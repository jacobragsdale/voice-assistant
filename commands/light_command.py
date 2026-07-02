import asyncio
import json
import os
import random
import threading
from typing import Dict, Any, Optional, Tuple

from openai import OpenAI
from kasa import Discover
from kasa.iot import IotBulb

from commands.base_command import BaseCommand


class LightCommand(BaseCommand):
    def __init__(self):
        super().__init__(
            name="light",
            description="Control Kasa smart lights",
            parameters={
                "action": {"type": "string", "description": "Action to perform (on, off, toggle, status, color, party)", "default": "status"},
                "device": {"type": "string", "description": "Device name or IP address", "default": "all"},
                "brightness": {"type": "string", "description": "Brightness level (1-100)", "default": "100"},
                "color": {"type": "string", "description": "Color name or description (e.g., red, blue, warm white)", "default": ""}
            },
            example_queries=[
                {"query": "turn on the lights", "parameters": {"action": "on"}},
                {"query": "set brightness to 50 percent", "parameters": {"action": "on", "brightness": "50"}},
                {"query": "toggle the bedroom light", "parameters": {"action": "toggle", "device": "bedroom"}},
                {"query": "are the lights on", "parameters": {"action": "status"}},
                {"query": "change the lights to blue", "parameters": {"action": "color", "color": "blue"}},
                {"query": "set living room light to warm white", "parameters": {"action": "color", "device": "living room", "color": "warm white"}},
                {"query": "turn on party mode", "parameters": {"action": "party"}},
                {"query": "start party mode", "parameters": {"action": "party"}}
            ]
        )
        self.devices: Dict[str, IotBulb] = {}
        self.devices_cache_file = "devices_cache.json"
        self.device_data_cache = {}
        self._load_devices_from_cache()
        self.color_cache_file = "color_cache.json"
        self.color_cache: Dict[str, Tuple[int, int, int]] = {}
        self._load_color_cache()
        self._loop = None
        self.party_thread = None
        self.party_active = False
        party_colors_path = os.path.join(os.path.dirname(__file__), 'party_colors.json')
        with open(party_colors_path, "r") as f:
            self.party_colors = json.load(f)

    async def execute(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        parameters = parameters or {}
        action = parameters.get("action", "status").lower()
        brightness_int = self._sanitize_brightness(parameters.get("brightness", "100"))
        color_description = parameters.get("color", "")
        color_hsv = None
        if action == "color" and color_description:
            color_hsv = self._convert_color_description_to_hsv(color_description)
            action = "color" if color_hsv else "on"
        
        if action == "party":
            await self._handle_party_mode(brightness_int)
        else:
            await self._run_command_in_loop(action, brightness_int, color_hsv)

    def _sanitize_brightness(self, brightness: str) -> int:
        try:
            b = int(brightness)
            return b if 1 <= b <= 100 else 100
        except Exception:
            return 100

    async def _run_command_in_loop(self, action: str, brightness: int, color_hsv: Optional[Tuple[int, int, int]] = None) -> None:
        try:
            await self._execute_light_command(action, color_hsv, brightness)
        except Exception as e:
            print(f"Error executing command: {e}")

    def _load_devices_from_cache(self) -> None:
        try:
            if os.path.exists(self.devices_cache_file):
                # Add a simple button
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
                    "device_type": "bulb" if isinstance(device, IotBulb) else "other"
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
            client = OpenAI(api_key=api_key)
            system_prompt = (
                "You are a color interpretation assistant. Your task is to convert natural language color descriptions into HSV color values. "
                "Respond with ONLY a valid JSON object containing the HSV values in this format: {\"h\": 0, \"s\": 100, \"v\": 100}"
            )
            response = client.chat.completions.create(
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

    async def set_devices(self):
        if self.device_data_cache:
            print("Using cached device information...")
            tasks = [
                self._connect_to_cached_devices(ip)
                for ip, info in self.device_data_cache.items()
            ]
            await asyncio.gather(*tasks)
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

    async def _execute_light_command(self, action: str, color_hsv, brightness: int) -> None:
        if not self.devices:
            await self.set_devices()

        async def run_command(dev):
            try:
                await dev.update()
                dev_alias = getattr(dev, "alias", "Unknown device")
                if action == "on":
                    if isinstance(dev, IotBulb):
                        await dev.set_brightness(brightness)
                    await dev.turn_on()
                    print(f"Turned on {dev_alias}" + (f" at {brightness}% brightness" if isinstance(dev, IotBulb) else ""))
                elif action == "off":
                    await dev.turn_off()
                    print(f"Turned off {dev_alias}")
                elif action == "toggle":
                    if dev.is_on:
                        await dev.turn_off()
                        print(f"Toggled {dev_alias} off")
                    else:
                        if isinstance(dev, IotBulb):
                            await dev.set_brightness(brightness)
                        await dev.turn_on()
                        print(f"Toggled {dev_alias} on" + (f" at {brightness}% brightness" if isinstance(dev, IotBulb) else ""))
                elif action == "color" and color_hsv is not None:
                    if isinstance(dev, IotBulb) and hasattr(dev, "set_hsv"):
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

        tasks = [run_command(dev) for dev in list(self.devices.values())]
        await asyncio.gather(*tasks)

    async def _connect_to_cached_devices(self, ip) -> None:
        try:
            device = IotBulb(ip)
            await device.update()
            self.devices[ip] = device
            print(f"Connected to {getattr(device, 'alias', 'Unknown device')} ({ip})")
        except Exception as e:
            print(f"Error connecting to cached device at {ip}: {e}")

    async def _handle_party_mode(self, brightness: int) -> None:
        """Handle the party mode command - start or stop party mode"""
        if self.party_active:
            self.party_active = False
            print("Party mode stopped")
        else:
            self.party_active = True
            await self._party_mode_thread(brightness)

    async def _party_mode_thread(self, brightness: int) -> None:
        """Thread function for continuously changing light colors"""
        try:
            while self.party_active:
                # Get random color from color cache
                if not self.party_colors:
                    print("No colors available in color cache")
                    self.party_active = False
                    break

                current_colors = list(self.party_colors.keys())
                random.shuffle(current_colors)
                random_color = current_colors[0]
                color_hsv = self.party_colors.get(random_color)
                
                if color_hsv:
                    print(f"Party mode: Changing to {random_color}")
                    await self._execute_light_command("color", color_hsv, brightness)
                await asyncio.sleep(5.0)
                
        except Exception as e:
            print(f"Error in party mode: {e}")
            self.party_active = False


if __name__ == '__main__':
    parameters = {'action': 'party'}
    light_command = LightCommand()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(light_command.execute(parameters))
        loop.run_forever()
    except KeyboardInterrupt:
        loop.close()
