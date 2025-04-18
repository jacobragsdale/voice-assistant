import asyncio
import json
import openai
import os
from typing import Dict, Any, Optional, List, Tuple
from commands.base_command import BaseCommand
from kasa import SmartBulb, SmartDevice, Discover

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
                {"query": "set living room light to warm orange", "parameters": {"action": "color", "device": "living room", "color": "warm orange"}}
            ]
        )
        # Cache for discovered devices
        self.devices: Dict[str, SmartDevice] = {}
        self.device_aliases: Dict[str, str] = {
            "living room": "192.168.1.100",  # Example mapping, replace with your actual devices
            "bedroom": "192.168.1.101",
            "kitchen": "192.168.1.102"
        }
        self.devices_cache_file = "devices_cache.json"
        self.device_data_cache = {}
        # Try loading devices from cache
        self._load_devices_from_cache()
        # Create an event loop that will be reused instead of creating a new one for each command
        self._loop = None
    
    def execute(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        if parameters is None:
            parameters = {}
        
        action = parameters.get('action', 'status').lower()
        device_name = parameters.get('device', 'all').lower()
        brightness = parameters.get('brightness', '100')
        color_description = parameters.get('color', '')
        
        try:
            brightness_int = int(brightness)
            if brightness_int < 1 or brightness_int > 100:
                brightness_int = 100
        except ValueError:
            brightness_int = 100
        
        # If action is color, convert color description to RGB
        color_rgb = None
        if action == "color" and color_description:
            color_rgb = self._convert_color_description_to_rgb(color_description)
            if color_rgb:
                print(f"Converting '{color_description}' to RGB: {color_rgb}")
                # Ensure action is set to color
                action = "color"
            else:
                print(f"Could not interpret color: '{color_description}'")
                action = "on"  # Default to turning on the light if color can't be determined
        
        # Run the async operation in a persistent event loop
        self._run_command_in_loop(action, device_name, brightness_int, color_rgb)
    
    def _run_command_in_loop(self, action: str, device_name: str, brightness: int, color_rgb: Optional[Tuple[int, int, int]] = None) -> None:
        """Run the command in a persistent event loop to avoid the 'loop closed' error."""
        try:
            # Create a new event loop if one doesn't exist or is closed
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            # Run the command in the existing loop
            self._loop.run_until_complete(self._execute_light_command(action, device_name, brightness, color_rgb))
            
        except Exception as e:
            print(f"Error executing command: {e}")
            # If we get an error, reset the loop for next time
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._loop = None
    
    def _load_devices_from_cache(self) -> None:
        """Load device cache from JSON file if available."""
        try:
            if os.path.exists(self.devices_cache_file):
                with open(self.devices_cache_file, 'r') as f:
                    self.device_data_cache = json.load(f)
                print(f"Loaded {len(self.device_data_cache)} devices from cache.")
        except Exception as e:
            print(f"Error loading device cache: {e}")
            # Ensure we have an empty cache if loading fails
            self.device_data_cache = {}
    
    def _save_devices_to_cache(self) -> None:
        """Save discovered devices to a JSON file for future use."""
        try:
            # Convert device data to a serializable format
            serialized_data = {}
            for ip, device in self.devices.items():
                # We can only store string data, so save key device properties
                device_info = {
                    "ip": ip,
                    "alias": getattr(device, "alias", "Unknown Device"),
                    "model": getattr(device, "model", "Unknown Model"),
                    "device_type": "bulb" if isinstance(device, SmartBulb) else "other",
                }
                serialized_data[ip] = device_info
            
            # Save to file
            with open(self.devices_cache_file, 'w') as f:
                json.dump(serialized_data, f, indent=2)
            
            print(f"Saved {len(serialized_data)} devices to cache.")
            self.device_data_cache = serialized_data
        except Exception as e:
            print(f"Error saving device cache: {e}")
    
    def _convert_color_description_to_rgb(self, color_description: str) -> Optional[Tuple[int, int, int]]:
        """
        Use OpenAI to convert a color description to RGB values.
        
        Args:
            color_description: Natural language description of a color (e.g., "deep blue", "warm orange")
            
        Returns:
            Tuple of (r, g, b) values or None if conversion failed
        """
        try:
            # For common colors, use predefined RGB values to avoid API call
            common_colors = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "yellow": (255, 255, 0),
                "purple": (128, 0, 128),
                "orange": (255, 165, 0),
                "pink": (255, 192, 203),
                "white": (255, 255, 255),
                "warm white": (255, 244, 229),
                "cool white": (212, 235, 255),
            }
            
            # Check if it's a common color
            for color_name, rgb in common_colors.items():
                if color_description.lower() == color_name:
                    return rgb
            
            # Use OpenAI to interpret more complex color descriptions
            # Get API key from environment
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print("OpenAI API key not found. Cannot interpret complex colors.")
                return None
                
            openai.api_key = api_key
            
            system_prompt = """You are a color interpretation assistant. 
            Your task is to convert natural language color descriptions into RGB color values.
            Respond with ONLY a valid JSON object containing the RGB values in this format:
            {"r": 255, "g": 0, "b": 0}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this color description to RGB values: {color_description}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if "r" in result and "g" in result and "b" in result:
                return (result["r"], result["g"], result["b"])
            
            return None
            
        except Exception as e:
            print(f"Error converting color: {e}")
            return None
    
    async def _execute_light_command(self, action: str, device_name: str, brightness: int, color_rgb: Optional[Tuple[int, int, int]] = None) -> None:
        """Execute the light command asynchronously."""
        # Discover devices if we haven't already or use cached device information
        if not self.devices:
            if self.device_data_cache:
                # Use cached device information to create a quick connection
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
                    
                    # Save the newly discovered devices to cache
                    self._save_devices_to_cache()
                except Exception as e:
                    print(f"Error discovering devices: {e}")
                    return
        
        # Determine which device(s) to control
        target_devices: List[SmartDevice] = []
        
        if device_name == "all":
            target_devices = list(self.devices.values())
        else:
            # Try to find by alias first
            device_ip = self.device_aliases.get(device_name)
            if device_ip and device_ip in self.devices:
                target_devices = [self.devices[device_ip]]
            else:
                # Try to find by partial name match
                for dev in self.devices.values():
                    try:
                        await dev.update()  # Refresh device info
                        if hasattr(dev, 'alias') and device_name.lower() in dev.alias.lower():
                            target_devices.append(dev)
                    except Exception as e:
                        print(f"Error updating device info: {e}")
                        # Try to reconnect if device update fails
                        continue
        
        if not target_devices:
            print(f"No devices found matching '{device_name}'")
            return
        
        # Execute the requested action on all target devices
        for device in target_devices:
            try:
                await device.update()  # Refresh device info
                device_name = getattr(device, 'alias', 'Unknown device')
                
                if action == "on":
                    if isinstance(device, SmartBulb):
                        await device.set_brightness(brightness)
                    await device.turn_on()
                    print(f"Turned on {device_name}" + (f" at {brightness}% brightness" if isinstance(device, SmartBulb) else ""))
                
                elif action == "off":
                    await device.turn_off()
                    print(f"Turned off {device_name}")
                
                elif action == "toggle":
                    if device.is_on:
                        await device.turn_off()
                        print(f"Toggled {device_name} off")
                    else:
                        if isinstance(device, SmartBulb):
                            await device.set_brightness(brightness)
                        await device.turn_on()
                        print(f"Toggled {device_name} on" + (f" at {brightness}% brightness" if isinstance(device, SmartBulb) else ""))
                
                elif action == "color" and color_rgb is not None:
                    if isinstance(device, SmartBulb) and hasattr(device, 'set_hsv'):
                        # Convert RGB to HSV (Kasa bulbs use HSV)
                        hsv = self._rgb_to_hsv(*color_rgb)
                        await device.set_hsv(*hsv)
                        await device.set_brightness(brightness)
                        await device.turn_on()
                        print(f"Set {device_name} to color RGB{color_rgb} at {brightness}% brightness")
                    else:
                        print(f"Device {device_name} doesn't support color changes")
                
                elif action == "status":
                    state = "on" if device.is_on else "off"
                    brightness_str = f" at {device.brightness}% brightness" if hasattr(device, 'brightness') else ""
                    color_str = ""
                    if hasattr(device, 'hsv') and device.is_on:
                        color_str = f", color HSV: {device.hsv}"
                    print(f"{device_name} is {state}{brightness_str}{color_str}")
                
                else:
                    print(f"Unknown action: {action}")
            
            except Exception as e:
                print(f"Error controlling {getattr(device, 'alias', 'Unknown device')}: {e}")
                continue
    
    async def _connect_to_cached_devices(self) -> None:
        """Connect to devices using cached IP addresses."""
        for ip, device_info in self.device_data_cache.items():
            try:
                # Determine device type
                if device_info.get("device_type") == "bulb":
                    device = SmartBulb(ip)
                else:
                    device = SmartDevice(ip)
                
                # Connect to the device
                await device.update()
                self.devices[ip] = device
                print(f"Connected to {getattr(device, 'alias', 'Unknown device')} ({ip})")
            except Exception as e:
                print(f"Error connecting to cached device at {ip}: {e}")

    def _rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """
        Convert RGB color values to HSV (Hue, Saturation, Value) for Kasa bulbs.
        
        Args:
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
            
        Returns:
            Tuple of (hue, saturation, value) where:
            - hue is 0-360
            - saturation is 0-100
            - value is 0-100
        """
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
            
        s = 0 if mx == 0 else df/mx * 100
        v = mx * 100
        
        return (int(h), int(s), int(v)) 