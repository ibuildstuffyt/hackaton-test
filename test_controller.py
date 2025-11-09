"""
Test program to display DJI controller axis values
"""
import pygame
import sys

def main():
    pygame.init()
    pygame.joystick.init()
    
    # Check for controllers
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No controller detected!")
        print("Please connect your DJI controller via USB.")
        sys.exit(1)
    
    # Initialize first controller
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print("=" * 60)
    print(f"Controller: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print("=" * 60)
    print("\nMove your controller sticks to see the values change.")
    print("Press ESC or Ctrl+C to exit.\n")
    
    try:
        while True:
            pygame.event.pump()
            
            # Get all axis values
            axis_values = []
            for i in range(joystick.get_numaxes()):
                axis_values.append(joystick.get_axis(i))
            
            # Clear screen (simple approach - print newlines)
            print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
            
            print("=" * 60)
            print("DJI Controller - Channel Values")
            print("=" * 60)
            print(f"\nAxis 0 (Channel 0): {axis_values[0]:7.3f}  {'<-- Roll' if abs(axis_values[0]) > 0.05 else ''}")
            print(f"Axis 1 (Channel 1): {axis_values[1]:7.3f}  {'<-- Pitch' if abs(axis_values[1]) > 0.05 else ''}")
            print(f"Axis 2 (Channel 2): {axis_values[2]:7.3f}  {'<-- Throttle' if abs(axis_values[2]) > 0.05 else ''}")
            if len(axis_values) > 3:
                print(f"Axis 3 (Channel 3): {axis_values[3]:7.3f}  {'<-- Yaw' if abs(axis_values[3]) > 0.05 else ''}")
            if len(axis_values) > 4:
                print(f"Axis 4 (Channel 4): {axis_values[4]:7.3f}")
            
            print("\n" + "=" * 60)
            print("Current Mapping:")
            print("  Axis 0 = Roll")
            print("  Axis 1 = Pitch")
            print("  Axis 2 = Throttle")
            print("  Axis 3 = Yaw")
            print("=" * 60)
            
            # Check for ESC key
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit(0)
            
            pygame.time.wait(50)  # Update every 50ms
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()


