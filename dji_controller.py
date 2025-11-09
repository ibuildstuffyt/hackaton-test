"""
DJI Controller input handler
"""
import pygame
import numpy as np


class DJIController:
    """Handles DJI controller input via pygame"""
    
    def __init__(self):
        pygame.joystick.init()
        self.joystick = None
        self.connected = False
        self.initialize()
    
    def initialize(self):
        """Initialize controller connection"""
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
            print(f"DJI Controller connected: {self.joystick.get_name()}")
            print(f"Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")
        else:
            self.connected = False
            print("No controller detected")
    
    def get_throttle(self) -> float:
        """Get throttle input (Axis 2) - returns raw value"""
        if not self.connected or not self.joystick:
            return 0.0
        try:
            throttle = self.joystick.get_axis(2)
            return np.clip(throttle, -1.0, 1.0)
        except Exception as e:
            print(f"Error reading throttle: {e}")
            return 0.0
    
    def get_roll(self) -> float:
        """Get roll input (Axis 0) - returns raw value"""
        if not self.connected or not self.joystick:
            return 0.0
        try:
            roll = self.joystick.get_axis(0)
            return np.clip(roll, -1.0, 1.0)
        except Exception as e:
            print(f"Error reading roll: {e}")
            return 0.0
