# FPV Simulator with Marble World Integration

A First Person View (FPV) drone simulator that integrates with Marble World (marble.worldlabs.ai) and supports DJI controller input.

## Features

- 🎮 **DJI Controller Support**: Connect your DJI FPV controller via USB
- 🌍 **Marble World Integration**: Import and fly in worlds from marble.worldlabs.ai
- 🚁 **Realistic Physics**: Physics-based drone flight simulation
- 🎥 **FPV Camera**: First-person view camera that follows the drone
- 🎨 **3D Graphics**: OpenGL-based 3D rendering

## Requirements

- Python 3.8 or higher
- DJI FPV Controller (connected via USB)
- Marble World API key (optional, for importing worlds)

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key (optional):**
   ```bash
   cp .env.example .env
   # Edit .env and add your MARBLE_API_KEY
   ```

## Usage

1. **Connect your DJI controller:**
   - Connect your DJI FPV controller to your computer via USB
   - The controller should be recognized automatically

2. **Run the simulator:**
   ```bash
   python main.py
   ```

3. **Controls:**
   - **Left Stick Y-axis**: Throttle (up/down)
   - **Left Stick X-axis**: Yaw (rotate left/right)
   - **Right Stick Y-axis**: Pitch (tilt forward/backward)
   - **Right Stick X-axis**: Roll (tilt left/right)
   - **ESC**: Exit simulator
   - **C**: Calibrate controller (shows current axis values)

## Controller Calibration

If your controller inputs don't match the expected behavior:

1. Press **C** in the simulator to see current axis values
2. Check which axes correspond to which controls
3. You may need to adjust the axis mappings in `dji_controller.py`:
   - `get_throttle()`: Default uses axis 1
   - `get_yaw()`: Default uses axis 0
   - `get_pitch()`: Default uses axis 3
   - `get_roll()`: Default uses axis 2

## Marble World Integration

The simulator can import worlds from Marble World API:

1. Set your `MARBLE_API_KEY` in the `.env` file
2. The simulator will automatically fetch available worlds
3. Worlds are exported in GLTF format and loaded into the simulator

**Note**: Full GLTF parsing is not yet implemented. The current version uses a test environment. To add full GLTF support, you'll need to add a GLTF parser library (e.g., `pygltflib`).

## Project Structure

```
.
├── main.py              # Main entry point
├── marble_api.py        # Marble World API client
├── dji_controller.py    # DJI controller input handler
├── fpv_simulator.py     # 3D FPV simulator with physics
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Troubleshooting

### Controller not detected
- Make sure the controller is connected via USB
- On Mac, you may need a USB-C to USB-A adapter
- Try unplugging and reconnecting the controller
- Check if the controller is recognized by your OS (should appear as a game controller)

### Controller inputs are inverted or wrong
- Press **C** to calibrate and see which axes are active
- Adjust the axis mappings in `dji_controller.py` based on your controller model
- Some DJI controllers may have different axis layouts

### API connection issues
- Check your internet connection
- Verify your API key is correct in the `.env` file
- The simulator will work with a test environment even without API access

### Performance issues
- Lower the window resolution in `fpv_simulator.py` (change `width` and `height` in `FPVSimulator.__init__`)
- Reduce the complexity of the world mesh

## Future Enhancements

- [ ] Full GLTF/OBJ file parsing for Marble World imports
- [ ] More realistic drone physics
- [ ] Multiple camera views (FPV, third-person, etc.)
- [ ] Recording and playback of flights
- [ ] Multiplayer support
- [ ] Customizable drone parameters
- [ ] Weather effects
- [ ] Obstacle collision detection

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues and enhancement requests!




