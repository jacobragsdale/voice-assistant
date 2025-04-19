import asyncio
import json
import openai
import os
from typing import Dict, Any, Optional, List, Tuple
from commands.base_command import BaseCommand
from kasa import SmartBulb, SmartDevice, Discover

from tts import VoiceAssistant


class LightCommand(BaseCommand):
    def __init__(self):
        super().__init__(
            name="light",
            description="Control Kasa smart lights",
            parameters={
                "action": {"type": "string", "description": "Action to perform (on, off, toggle, status, color)", "default": "status"},
                "device": {"type": "string", "description": "Device name or IP address", "default": "all"},
                "brightness": {"type": "string", "description": "Brightness level (1-100)", "default": "100"},
                "color": {"type": "string", "description": "Color name or description (e.g., red, blue, warm white)", "default": ""}
            },
            example_queries=[
                {"query": "turn on the lights", "parameters": {"action": "on"}},
                {"query": "turn off the living room light", "parameters": {"action": "off", "device": "living room"}},
                {"query": "set brightness to 50 percent", "parameters": {"action": "on", "brightness": "50"}},
                {"query": "toggle the bedroom light", "parameters": {"action": "toggle", "device": "bedroom"}},
                {"query": "are the lights on", "parameters": {"action": "status"}},
                {"query": "change the lights to blue", "parameters": {"action": "color", "color": "blue"}},
                {"query": "set living room light to warm white", "parameters": {"action": "color", "device": "living room", "color": "warm white"}}
            ]
        )
        self.devices: Dict[str, SmartDevice] = {}
        self.device_aliases: Dict[str, str] = {
            "living room": "192.168.1.100",
            "bedroom": "192.168.1.101",
            "kitchen": "192.168.1.102"
        }
        self.devices_cache_file = "devices_cache.json"
        self.device_data_cache = {}
        self._load_devices_from_cache()
        self.color_cache_file = "color_cache.json"
        self.color_cache: Dict[str, Tuple[int, int, int]] = {}
        self._load_color_cache()
        self._loop = None

    def execute(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        parameters = parameters or {}
        action = parameters.get("action", "status").lower()
        device_name = parameters.get("device", "all").lower()
        brightness_int = self._sanitize_brightness(parameters.get("brightness", "100"))
        color_description = parameters.get("color", "")
        color_hsv = None
        if action == "color" and color_description:
            color_hsv = self._convert_color_description_to_hsv(color_description)
            action = "color" if color_hsv else "on"
        self._run_command_in_loop(action, device_name, brightness_int, color_hsv)

    def _sanitize_brightness(self, brightness: str) -> int:
        try:
            b = int(brightness)
            return b if 1 <= b <= 100 else 100
        except Exception:
            return 100

    def _run_command_in_loop(self, action: str, device_name: str, brightness: int, color_hsv: Optional[Tuple[int, int, int]] = None) -> None:
        try:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._execute_light_command(action, device_name, brightness, color_hsv))
        except Exception as e:
            print(f"Error executing command: {e}")
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    def _load_devices_from_cache(self) -> None:
        try:
            if os.path.exists(self.devices_cache_file):
                with open(self.devices_cache_file, "r") as f:
                    self.device_data_cache = json.load(f)
                print(f"Loaded {len(self.device_data_cache)} devices from cache.")
        except Exception as e:
            print(f"Error loading device cache: {e}")
            self.device_data_cache = {}

    def _save_devices_to_cache(self) -> None:
        try:
            serialized_data = {}
            for ip, device in self.devices.items():
                device_info = {
                    "ip": ip,
                    "alias": getattr(device, "alias", "Unknown Device"),
                    "model": getattr(device, "model", "Unknown Model"),
                    "device_type": "bulb" if isinstance(device, SmartBulb) else "other"
                }
                serialized_data[ip] = device_info
            with open(self.devices_cache_file, "w") as f:
                json.dump(serialized_data, f, indent=2)
            print(f"Saved {len(serialized_data)} devices to cache.")
            self.device_data_cache = serialized_data
        except Exception as e:
            print(f"Error saving device cache: {e}")

    def _load_color_cache(self) -> None:
        try:
            if os.path.exists(self.color_cache_file):
                with open(self.color_cache_file, "r") as f:
                    self.color_cache = json.load(f)
                print(f"Loaded {len(self.color_cache)} color mappings from cache.")
        except Exception as e:
            print(f"Error loading color cache: {e}")
            self.color_cache = {}

    def _save_color_cache(self) -> None:
        try:
            with open(self.color_cache_file, "w") as f:
                json.dump(self.color_cache, f, indent=2)
            print(f"Saved {len(self.color_cache)} color mappings to cache.")
        except Exception as e:
            print(f"Error saving color cache: {e}")

    def _convert_color_description_to_hsv(self, color_description: str) -> Optional[Tuple[int, int, int]]:
        try:
            key = color_description.lower()
            if key in self.color_cache:
                return tuple(self.color_cache[key])
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OpenAI API key not found. Cannot interpret complex colors.")
                return None
            openai.api_key = api_key
            system_prompt = (
                "You are a color interpretation assistant. Your task is to convert natural language color descriptions into HSV color values. "
                "Respond with ONLY a valid JSON object containing the HSV values in this format: {\"h\": 0, \"s\": 100, \"v\": 100}"
            )
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this color description to HSV values: {color_description}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            if "h" in result and "s" in result and "v" in result:
                hsv = (result["h"], result["s"], result["v"])
                self.color_cache[key] = hsv
                self._save_color_cache()
                return hsv
            return None
        except Exception as e:
            print(f"Error converting color: {e}")
            return None

    async def _execute_light_command(self, action: str, device_name: str, brightness: int, color_hsv: Optional[Tuple[int, int, int]] = None) -> None:

        if not self.devices:
            if self.device_data_cache:
                print("Using cached device information...")
                await self._connect_to_cached_devices()
            else:
                print("Discovering devices...")
                try:
                    discovered_devices = await Discover.discover()
                    if not discovered_devices:
                        print("No Kasa devices found on the network.")
                        return
                    self.devices = discovered_devices
                    print(f"Found {len(self.devices)} device(s)")
                    self._save_devices_to_cache()
                except Exception as e:
                    print(f"Error discovering devices: {e}")
                    return
        target_devices: List[SmartDevice] = []
        if device_name == "all":
            target_devices = list(self.devices.values())
        else:
            device_ip = self.device_aliases.get(device_name)
            if device_ip and device_ip in self.devices:
                target_devices = [self.devices[device_ip]]
            else:
                for dev in self.devices.values():
                    try:
                        await dev.update()
                        if hasattr(dev, "alias") and device_name in dev.alias.lower():
                            target_devices.append(dev)
                    except Exception as e:
                        print(f"Error updating device info: {e}")
                        continue
        if not target_devices:
            print(f"No devices found matching '{device_name}'")
            return

        for dev in target_devices:
            try:
                await dev.update()
                dev_alias = getattr(dev, "alias", "Unknown device")
                if action == "on":
                    if isinstance(dev, SmartBulb):
                        await dev.set_brightness(brightness)
                    await dev.turn_on()
                    print(f"Turned on {dev_alias}" + (f" at {brightness}% brightness" if isinstance(dev, SmartBulb) else ""))
                elif action == "off":
                    await dev.turn_off()
                    print(f"Turned off {dev_alias}")
                elif action == "toggle":
                    if dev.is_on:
                        await dev.turn_off()
                        print(f"Toggled {dev_alias} off")
                    else:
                        if isinstance(dev, SmartBulb):
                            await dev.set_brightness(brightness)
                        await dev.turn_on()
                        print(f"Toggled {dev_alias} on" + (f" at {brightness}% brightness" if isinstance(dev, SmartBulb) else ""))
                elif action == "color" and color_hsv is not None:
                    if isinstance(dev, SmartBulb) and hasattr(dev, "set_hsv"):
                        await dev.set_hsv(*color_hsv)
                        await dev.set_brightness(brightness)
                        await dev.turn_on()
                        print(f"Set {dev_alias} to color HSV{color_hsv} at {brightness}% brightness")
                    else:
                        print(f"Device {dev_alias} doesn't support color changes")
                elif action == "status":
                    state = "on" if dev.is_on else "off"
                    brightness_str = f" at {dev.brightness}% brightness" if hasattr(dev, "brightness") else ""
                    color_str = f", color HSV: {dev.hsv}" if hasattr(dev, "hsv") and dev.is_on else ""
                    print(f"{dev_alias} is {state}{brightness_str}{color_str}")
                else:
                    print(f"Unknown action: {action}")
            except Exception as e:
                print(f"Error controlling {getattr(dev, 'alias', 'Unknown device')}: {e}")
                continue

    async def _connect_to_cached_devices(self) -> None:
        for ip, info in self.device_data_cache.items():
            try:
                device = SmartBulb(ip) if info.get("device_type") == "bulb" else SmartDevice(ip)
                await device.update()
                self.devices[ip] = device
                print(f"Connected to {getattr(device, 'alias', 'Unknown device')} ({ip})")
            except Exception as e:
                print(f"Error connecting to cached device at {ip}: {e}")