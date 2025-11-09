# Quick Start Guide - Step by Step

Follow these steps to get your FPV simulator running with your DJI controller.

## Step 1: Connect Your DJI Controller

1. **Power on your DJI controller** (if it has a power button)
2. **Connect the controller to your Mac** using a USB cable:
   - If your controller has USB-C, use a USB-C to USB-A cable (or USB-C to USB-C if your Mac supports it)
   - On older Macs, you may need a USB-A to USB-C adapter
3. **Wait a few seconds** for your Mac to recognize the controller
4. You should see a notification or the controller should appear in System Settings > Game Controllers (optional check)

## Step 2: Install Python Dependencies

Open Terminal and navigate to the project folder:

```bash
cd "/Users/johnxu/Desktop/hackaton test"
```

Install the required packages:

```bash
pip3 install -r requirements.txt
```

**If you get permission errors**, try:
```bash
pip3 install --user -r requirements.txt
```

**If pip3 is not found**, try:
```bash
python3 -m pip install -r requirements.txt
```

Wait for all packages to install. This may take a minute or two.

## Step 3: Test Your Controller Connection

Before running the full simulator, let's verify your controller works:

```bash
python3 test_controller.py
```

**What to expect:**
- A window should open showing your controller's axis values and buttons
- Move the sticks and press buttons - you should see the values change
- The green bars should move as you move the sticks
- Press **ESC** or close the window when done

**If the controller is NOT detected:**
- Make sure it's connected via USB
- Try unplugging and reconnecting
- Check if it appears in System Settings > Game Controllers
- Some controllers need to be in a specific mode (check your DJI controller manual)

**Note the axis numbers** that respond to:
- Left stick up/down (throttle)
- Left stick left/right (yaw)
- Right stick up/down (pitch)
- Right stick left/right (roll)

## Step 4: (Optional) Set Up Marble World API

If you want to import worlds from Marble World:

1. Create a `.env` file in the project folder:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file in a text editor and add your API key:
   ```
   MARBLE_API_KEY=your_actual_api_key_here
   ```

3. Save the file

**Note:** The simulator will work without an API key - it will just use a test environment.

## Step 5: Run the FPV Simulator

Now you're ready to fly! Run:

```bash
python3 main.py
```

**What to expect:**
- A 3D window should open showing a test environment
- You'll see a red cube (the drone) floating in the air
- The camera view is from the drone's perspective (FPV)

## Step 6: Control the Drone

**Default Controls:**
- **Left Stick Y-axis (up/down)**: Throttle - controls altitude
- **Left Stick X-axis (left/right)**: Yaw - rotates the drone
- **Right Stick Y-axis (up/down)**: Pitch - tilts forward/backward
- **Right Stick X-axis (left/right)**: Roll - tilts left/right

**Keyboard Controls:**
- **ESC**: Exit the simulator
- **C**: Calibrate controller (shows current axis values in terminal)

## Step 7: If Controls Don't Work Correctly

If the controls feel inverted or wrong:

1. Press **C** in the simulator to see which axes are active
2. Note which axis number corresponds to which stick movement
3. Open `dji_controller.py` in a text editor
4. Find the functions `get_throttle()`, `get_yaw()`, `get_pitch()`, `get_roll()`
5. Change the axis numbers (0, 1, 2, 3) to match what you saw in the calibration

For example, if throttle should be axis 2 instead of axis 1:
```python
def get_throttle(self) -> float:
    throttle = -self.joystick.get_axis(2)  # Changed from 1 to 2
    return np.clip(throttle, -1.0, 1.0)
```

## Troubleshooting

### "No module named 'pygame'" or similar errors
- Make sure you ran `pip3 install -r requirements.txt`
- Try: `python3 -m pip install --upgrade pip` then install again

### Controller not detected
- Check USB connection
- Try a different USB port
- On Mac, some USB-C ports work better than others
- Make sure the controller is powered on

### Simulator window doesn't open
- Check Terminal for error messages
- Make sure you have OpenGL support (most Macs do)
- Try updating your graphics drivers

### Controls are inverted
- Press **C** to calibrate and see axis values
- Adjust axis mappings in `dji_controller.py` as described above

### Simulator runs but drone doesn't move
- Make sure controller is connected
- Check that you're moving the correct sticks
- Try pressing **C** to verify controller input is being read

## Need Help?

Check the main `README.md` file for more detailed information about the project structure and features.




