"""
FPV Simulator with realistic quadcopter physics
Only throttle and roll enabled
"""
import pygame
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from dji_controller import DJIController


class QuadcopterPhysics:
    """Realistic quadcopter physics simulation"""
    
    def __init__(self):
        # Physical properties
        self.mass = 0.5  # kg
        self.gravity = 9.81  # m/s^2
        self.max_thrust_per_motor = 2.0 * self.mass * self.gravity / 4.0  # N per motor
        self.motor_arm_length = 0.15  # m
        
        # Inertia tensor (simplified as diagonal) - increased to reduce dangling mass feel
        self.Ixx = 0.02  # kg*m^2 (increased from 0.01)
        self.Iyy = 0.02  # kg*m^2 (increased from 0.01)
        self.Izz = 0.04  # kg*m^2 (increased from 0.02)
        
        # Drag coefficients
        self.linear_drag = 0.1
        self.angular_drag = 5.0  # Increased to 5x to reduce momentum buildup by 5x
        
        # State
        self.position = np.array([0.0, 0.2, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # quaternion (w, x, y, z)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # (wx, wy, wz)
        self.motor_speeds = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quaternion_rotate_vector(self, q, v):
        """Rotate vector by quaternion"""
        w, x, y, z = q
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        q_inv = np.array([w, -x, -y, -z])
        temp = self.quaternion_multiply(q, v_quat)
        result_quat = self.quaternion_multiply(temp, q_inv)
        return result_quat[1:4]
    
    def set_motor_speeds(self, throttle: float, pitch: float, roll: float, yaw: float):
        """Set motor speeds - throttle, pitch, roll, and yaw"""
        # Store inputs for reverse detection
        self.last_pitch_input = getattr(self, 'last_pitch_input', 0.0)
        self.last_roll_input = getattr(self, 'last_roll_input', 0.0)
        self.last_yaw_input = getattr(self, 'last_yaw_input', 0.0)
        
        # Deadzones
        deadzone = 0.05
        if abs(pitch) < deadzone:
            pitch = 0.0
        if abs(roll) < deadzone:
            roll = 0.0
        if abs(yaw) < deadzone:
            yaw = 0.0
        
        # Detect direction reversal for instant response
        self.pitch_reversed = (pitch > 0 and self.last_pitch_input < 0) or (pitch < 0 and self.last_pitch_input > 0)
        self.roll_reversed = (roll > 0 and self.last_roll_input < 0) or (roll < 0 and self.last_roll_input > 0)
        self.yaw_reversed = (yaw > 0 and self.last_yaw_input < 0) or (yaw < 0 and self.last_yaw_input > 0)
        
        self.last_pitch_input = pitch
        self.last_roll_input = roll
        self.last_yaw_input = yaw
        
        # Motor mixing
        # Motors: m1=front-left[0], m2=front-right[1], m3=back-left[2], m4=back-right[3]
        # Pitch: forward pitch (positive) = tilt forward = increase front motors (m1, m2), decrease back motors (m3, m4)
        # Roll: right roll (positive) = tilt right = increase right motors (m2, m4), decrease left motors (m1, m3)
        # Yaw: right yaw (positive) = rotate right = increase diagonal motors (m1, m4), decrease (m2, m3)
        base = throttle
        pitch_gain = 0.06  # Reduced sensitivity
        roll_gain = 0.06  # Reduced sensitivity
        yaw_gain = 0.5  # 5x more sensitive (was 0.1, now 0.5)
        m1 = base + pitch * pitch_gain - roll * roll_gain + yaw * yaw_gain  # front-left
        m2 = base + pitch * pitch_gain + roll * roll_gain - yaw * yaw_gain  # front-right
        m3 = base - pitch * pitch_gain - roll * roll_gain - yaw * yaw_gain  # back-left
        m4 = base - pitch * pitch_gain + roll * roll_gain + yaw * yaw_gain  # back-right
        
        self.motor_speeds = np.clip([m1, m2, m3, m4], 0.0, 1.0)
    
    def update(self, dt: float):
        """Update physics"""
        total_thrust = np.sum(self.motor_speeds) * self.max_thrust_per_motor
        
        # Torques
        # Motors: m1=front-left[0], m2=front-right[1], m3=back-left[2], m4=back-right[3]
        # Pitch: front motors (m1, m2) - back motors (m3, m4)
        pitch_torque = (self.motor_speeds[0] + self.motor_speeds[1] - 
                       self.motor_speeds[2] - self.motor_speeds[3]) * self.max_thrust_per_motor * self.motor_arm_length
        # Roll: right motors (m2, m4) - left motors (m1, m3)
        roll_torque = (self.motor_speeds[1] + self.motor_speeds[3] - 
                      self.motor_speeds[0] - self.motor_speeds[2]) * self.max_thrust_per_motor * self.motor_arm_length
        # Yaw: diagonal motors (m1, m4) - (m2, m3)
        yaw_torque = (self.motor_speeds[0] + self.motor_speeds[3] - 
                     self.motor_speeds[1] - self.motor_speeds[2]) * self.max_thrust_per_motor * self.motor_arm_length * 0.5
        
        # Update angular velocity
        # Pitch rotates around X-axis (forward/back tilt)
        # If input reversed direction, aggressively reverse opposing angular velocity (10x more aggressive)
        if hasattr(self, 'pitch_reversed') and self.pitch_reversed:
            if (pitch_torque > 0 and self.angular_velocity[0] < 0) or (pitch_torque < 0 and self.angular_velocity[0] > 0):
                # 10x more aggressive: zero out and apply strong correction in opposite direction
                self.angular_velocity[0] = 0.0
                # Apply 10x stronger correction torque
                self.angular_velocity[0] += (pitch_torque / self.Ixx) * dt * 10.0
        else:
            self.angular_velocity[0] += (pitch_torque / self.Ixx) * dt   # X-axis = pitch
        
        # Yaw rotates around Y-axis (rotation) - REVERSED
        if hasattr(self, 'yaw_reversed') and self.yaw_reversed:
            if (-yaw_torque > 0 and self.angular_velocity[1] < 0) or (-yaw_torque < 0 and self.angular_velocity[1] > 0):
                # 10x more aggressive: zero out and apply strong correction in opposite direction
                self.angular_velocity[1] = 0.0
                # Apply 10x stronger correction torque
                self.angular_velocity[1] += (-yaw_torque / self.Iyy) * dt * 10.0
        else:
            self.angular_velocity[1] += (-yaw_torque / self.Iyy) * dt   # Y-axis = yaw (reversed)
        
        # Roll rotates around Z-axis (left/right tilt) - REVERSED
        if hasattr(self, 'roll_reversed') and self.roll_reversed:
            if (-roll_torque > 0 and self.angular_velocity[2] < 0) or (-roll_torque < 0 and self.angular_velocity[2] > 0):
                # 10x more aggressive: zero out and apply strong correction in opposite direction
                self.angular_velocity[2] = 0.0
                # Apply 10x stronger correction torque
                self.angular_velocity[2] += (-roll_torque / self.Izz) * dt * 10.0
        else:
            self.angular_velocity[2] += (-roll_torque / self.Izz) * dt   # Z-axis = roll (reversed)
        
        self.angular_velocity *= (1.0 - self.angular_drag * dt)
        
        # Update orientation
        w, x, y, z = self.orientation
        wx, wy, wz = self.angular_velocity
        q_dot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
            w*wx + y*wz - z*wy,
            w*wy - x*wz + z*wx,
            w*wz + x*wy - y*wx
        ])
        self.orientation += q_dot * dt
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm
        
        # Thrust vector in world frame
        up_local = np.array([0.0, 1.0, 0.0])
        up_world = self.quaternion_rotate_vector(self.orientation, up_local)
        thrust_force = up_world * total_thrust
        
        # Gravity
        gravity_force = np.array([0.0, -self.mass * self.gravity, 0.0])
        
        # Total force
        total_force = thrust_force + gravity_force
        
        # Drag
        drag_force = -self.velocity * self.linear_drag
        
        # Update velocity
        acceleration = (total_force + drag_force) / self.mass
        self.velocity += acceleration * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Ground collision
        if self.position[1] < 0.0:
            self.position[1] = 0.0
            self.velocity[1] = 0.0
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to 4x4 rotation matrix for OpenGL"""
        w, x, y, z = q
        # Normalize
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Build 4x4 matrix
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w), 0],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w), 0],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)


class FPVSimulator:
    """Main FPV simulator"""
    
    def __init__(self, window_mode=1):
        """
        Initialize FPV Simulator
        window_mode: 1 for controller view (Window 1), 2 for 360 view (Window 2)
        """
        import os
        
        pygame.init()
        self.window_mode = window_mode
        self.width = 1280
        self.height = 720
        
        # Position windows side by side on macOS
        if window_mode == 1:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '50,100'
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
            pygame.display.set_caption("1 - Controller View")
        else:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f'{self.width + 100},100'
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL)
            pygame.display.set_caption("2 - 360° View")
        
        # Initialize OpenGL
        self.init_opengl()
        
        self.controller = DJIController()
        self.physics = QuadcopterPhysics()
        
        # Second camera rotation (independent WASD-controlled camera)
        self.camera2_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Quaternion (w, x, y, z)
        
        self.world_vertices = []
        self.world_faces = []
        self.hoops = []  # List of hoops: [(x, z, y_height, radius), ...]
        
        self.clock = pygame.time.Clock()
        self.dt = 0.016
        
        # Shared state file for multi-window sync
        self.state_file = "/tmp/fpv_sim_state.npy"
    
    def init_opengl(self):
        """Initialize OpenGL state once at startup"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        glClearColor(0.5, 0.7, 1.0, 1.0)
        
        # Set up projection matrix once
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Better lighting setup for depth perception
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 5.0, 1.0])  # Positional light
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
    def load_world_mesh(self, vertices: list, faces: list):
        self.world_vertices = vertices
        self.world_faces = faces
        # Reset pre-calculated faces when new world is loaded
        self._ground_faces = None
        self._block_faces = None
    
    def load_hoops(self, hoops: list):
        """Load hoop positions for the course"""
        self.hoops = hoops
    
    def reset_drone(self):
        self.physics.position = np.array([0.0, 0.2, 0.0], dtype=np.float32)
        self.physics.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.physics.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.physics.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.physics.motor_speeds = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Reset camera2 rotation to match drone orientation
        self.camera2_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        print("Drone reset")
    
    def update_camera(self):
        """Update FPV camera - 0 degree angle, follows drone orientation exactly"""
        glLoadIdentity()
        # Camera is at drone position with small offset forward
        camera_offset_local = np.array([0.0, 0.0, 0.1])  # Small forward offset
        camera_pos = self.physics.position + self.physics.quaternion_rotate_vector(
            self.physics.orientation, camera_offset_local
        )
        # Forward direction in drone's local frame (0 degree camera angle = straight ahead)
        forward_local = np.array([0.0, 0.0, -1.0])  # Forward in drone frame
        forward_world = self.physics.quaternion_rotate_vector(
            self.physics.orientation, forward_local
        )
        look_at = camera_pos + forward_world * 10.0
        # Up direction in drone's local frame
        up_local = np.array([0.0, 1.0, 0.0])  # Up in drone frame
        up_world = self.physics.quaternion_rotate_vector(
            self.physics.orientation, up_local
        )
        gluLookAt(
            camera_pos[0], camera_pos[1], camera_pos[2],
            look_at[0], look_at[1], look_at[2],
            up_world[0], up_world[1], up_world[2]
        )
    
    def update_camera_secondary(self):
        """Update second FPV camera - uses independent camera2_rotation (WASD-controlled)"""
        glLoadIdentity()
        # Camera is at drone position with small offset forward (same position as primary camera)
        camera_offset_local = np.array([0.0, 0.0, 0.1])  # Small forward offset
        camera_pos = self.physics.position + self.physics.quaternion_rotate_vector(
            self.physics.orientation, camera_offset_local
        )
        
        # Combine drone orientation with camera2 relative offset
        # This makes camera2's "neutral" position always face forward relative to drone
        # Like a pilot's body facing the nose, but head can rotate independently
        combined_rotation = self.physics.quaternion_multiply(
            self.physics.orientation, 
            self.camera2_rotation
        )
        
        # Forward direction based on combined rotation
        forward_local = np.array([0.0, 0.0, -1.0])  # Forward in camera frame
        forward_world = self.physics.quaternion_rotate_vector(
            combined_rotation, forward_local
        )
        look_at = camera_pos + forward_world * 10.0
        
        # Up direction based on combined rotation
        up_local = np.array([0.0, 1.0, 0.0])  # Up in camera frame
        up_world = self.physics.quaternion_rotate_vector(
            combined_rotation, up_local
        )
        gluLookAt(
            camera_pos[0], camera_pos[1], camera_pos[2],
            look_at[0], look_at[1], look_at[2],
            up_world[0], up_world[1], up_world[2]
        )
    
    def update_camera2_rotation(self, keys):
        """Update camera2 rotation based on WASD keyboard input"""
        rotation_speed = 2.0 * self.dt  # Rotation speed per frame
        
        # W/S: Pitch camera up/down (look up/down) - rotate around X-axis
        if keys[pygame.K_w]:
            # Pitch down: rotate around X-axis (positive for looking down)
            pitch_rot = np.array([
                math.cos(rotation_speed / 2),
                math.sin(rotation_speed / 2),
                0.0,
                0.0
            ], dtype=np.float32)
            self.camera2_rotation = self.physics.quaternion_multiply(self.camera2_rotation, pitch_rot)
        
        if keys[pygame.K_s]:
            # Pitch up: rotate around X-axis (negative for looking up)
            pitch_rot = np.array([
                math.cos(-rotation_speed / 2),
                math.sin(-rotation_speed / 2),
                0.0,
                0.0
            ], dtype=np.float32)
            self.camera2_rotation = self.physics.quaternion_multiply(self.camera2_rotation, pitch_rot)
        
        # A/D: Yaw camera left/right (look left/right) - rotate around Y-axis
        if keys[pygame.K_a]:
            # Yaw left: rotate around Y-axis (positive for left)
            yaw_rot = np.array([
                math.cos(rotation_speed / 2),
                0.0,
                math.sin(rotation_speed / 2),
                0.0
            ], dtype=np.float32)
            self.camera2_rotation = self.physics.quaternion_multiply(self.camera2_rotation, yaw_rot)
        
        if keys[pygame.K_d]:
            # Yaw right: rotate around Y-axis (negative for right)
            yaw_rot = np.array([
                math.cos(-rotation_speed / 2),
                0.0,
                math.sin(-rotation_speed / 2),
                0.0
            ], dtype=np.float32)
            self.camera2_rotation = self.physics.quaternion_multiply(self.camera2_rotation, yaw_rot)
        
        # Normalize quaternion to prevent drift
        norm = np.linalg.norm(self.camera2_rotation)
        if norm > 0:
            self.camera2_rotation /= norm
    
    def draw_drone(self):
        glPushMatrix()
        glTranslatef(self.physics.position[0], self.physics.position[1], self.physics.position[2])
        rot_matrix = self.physics.quaternion_to_rotation_matrix(self.physics.orientation)
        glMultMatrixf(rot_matrix.T)
        
        size = 0.05
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        # Front
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(-size, size, -size)
        # Back
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, -size, size)
        # Top
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        # Bottom
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, -size, -size)
        # Right
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        # Left
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, -size, size)
        glEnd()
        
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        arm_length = 0.13
        glVertex3f(0, 0, 0)
        glVertex3f(-arm_length, 0, -arm_length)
        glVertex3f(0, 0, 0)
        glVertex3f(arm_length, 0, -arm_length)
        glVertex3f(0, 0, 0)
        glVertex3f(-arm_length, 0, arm_length)
        glVertex3f(0, 0, 0)
        glVertex3f(arm_length, 0, arm_length)
        glEnd()
        
        glPopMatrix()
    
    def draw_world(self):
        if not self.world_vertices or not self.world_faces:
            return
        
        # Pre-calculate which faces are ground vs blocks for performance
        if not hasattr(self, '_ground_faces') or self._ground_faces is None:
            self._ground_faces = []
            self._block_faces = []
            for i, face in enumerate(self.world_faces):
                if len(face) >= 3:
                    is_ground = True
                    for idx in face[:3]:
                        if 0 <= idx < len(self.world_vertices):
                            if abs(self.world_vertices[idx][1]) > 0.01:
                                is_ground = False
                                break
                    if is_ground:
                        self._ground_faces.append(face)
                    else:
                        self._block_faces.append(face)
        
        # Draw ground with grid pattern for depth perception
        glDisable(GL_LIGHTING)
        glColor3f(0.4, 0.5, 0.4)  # Light green ground
        glBegin(GL_TRIANGLES)
        for face in self._ground_faces:
            for idx in face:
                if 0 <= idx < len(self.world_vertices):
                    v = self.world_vertices[idx]
                    glVertex3f(v[0], v[1], v[2])
        glEnd()
        
        # Draw blocks with lighting and different colors (optimized)
        glEnable(GL_LIGHTING)
        glBegin(GL_TRIANGLES)
        for face in self._block_faces:
            if len(face) >= 3:
                # Calculate normal for this face
                v0 = np.array(self.world_vertices[face[0]])
                v1 = np.array(self.world_vertices[face[1]])
                v2 = np.array(self.world_vertices[face[2]])
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                    glNormal3f(normal[0], normal[1], normal[2])
                
                # Color based on average height of face
                avg_height = (v0[1] + v1[1] + v2[1]) / 3.0
                height_factor = min(avg_height / 2.0, 1.0)
                color_r = 0.5 + height_factor * 0.3
                color_g = 0.3 + height_factor * 0.3
                color_b = 0.2 + height_factor * 0.2
                glColor3f(color_r, color_g, color_b)
                
                # Draw the triangle
                for idx in face:
                    if 0 <= idx < len(self.world_vertices):
                        v = self.world_vertices[idx]
                        glVertex3f(v[0], v[1], v[2])
        glEnd()
        
        # Draw grid lines on ground for depth perception (reduced density)
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.3, 0.2)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        grid_size = 100.0
        grid_spacing = 10.0  # Increased from 5.0 to reduce lines
        for i in range(int(-grid_size/grid_spacing), int(grid_size/grid_spacing) + 1):
            x = i * grid_spacing
            glVertex3f(x, 0.01, -grid_size)
            glVertex3f(x, 0.01, grid_size)
        for i in range(int(-grid_size/grid_spacing), int(grid_size/grid_spacing) + 1):
            z = i * grid_spacing
            glVertex3f(-grid_size, 0.01, z)
            glVertex3f(grid_size, 0.01, z)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def draw_hoops(self):
        """Draw hoops as wireframe rings"""
        if not self.hoops:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        
        for hoop in self.hoops:
            x, z, y_height, radius = hoop
            
            # Draw hoop as a circle in the XZ plane at height y_height
            segments = 32
            glColor3f(1.0, 0.5, 0.0)  # Orange color for visibility
            
            glBegin(GL_LINE_LOOP)
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                px = x + radius * math.cos(angle)
                pz = z + radius * math.sin(angle)
                glVertex3f(px, y_height, pz)
            glEnd()
            
            # Draw a few vertical support lines for better visibility
            glBegin(GL_LINES)
            for i in range(4):
                angle = 2.0 * math.pi * i / 4
                px = x + radius * math.cos(angle)
                pz = z + radius * math.sin(angle)
                glVertex3f(px, y_height - 0.5, pz)
                glVertex3f(px, y_height + 0.5, pz)
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _compute_crosshair_ndc_in_cam2(self, half_width: int):
        """Compute NDC position in camera-2 view where camera-1 forward points.
        Returns (x_ndc, y_ndc, clamped) where NDC is in [-1,1]."""
        # Field of view and aspect for right viewport
        aspect = (half_width / self.height) if self.height > 0 else 1.0
        fov_y = math.radians(70.0)
        fov_x = 2.0 * math.atan(aspect * math.tan(fov_y / 2.0))
        
        # World-space forward direction of camera-1 (drone orientation with 0deg tilt)
        forward1_world = self.physics.quaternion_rotate_vector(
            self.physics.orientation, np.array([0.0, 0.0, -1.0])
        )
        
        # Camera-2 absolute orientation = drone orientation * camera2 relative rotation
        q_cam2 = self.physics.quaternion_multiply(self.physics.orientation, self.camera2_rotation)
        # Inverse quaternion
        q_cam2_inv = np.array([q_cam2[0], -q_cam2[1], -q_cam2[2], -q_cam2[3]], dtype=np.float32)
        
        # Express camera-1 forward in camera-2 local space
        dir_cam2 = self.physics.quaternion_rotate_vector(q_cam2_inv, forward1_world)
        
        # Map to angles relative to camera-2 axes
        # yaw: left/right, pitch: up/down
        yaw = math.atan2(dir_cam2[0], -dir_cam2[2])
        horiz = math.sqrt(dir_cam2[0] * dir_cam2[0] + dir_cam2[2] * dir_cam2[2])
        pitch = math.atan2(dir_cam2[1], horiz)
        
        # Project angles to NDC using FOV
        x_ndc = math.tan(yaw) / math.tan(fov_x / 2.0) if fov_x > 0 else 0.0
        y_ndc = math.tan(pitch) / math.tan(fov_y / 2.0) if fov_y > 0 else 0.0
        
        # Clamp behavior: if behind camera-2, force to border; if out of frustum, clamp to border
        s = max(abs(x_ndc), abs(y_ndc))
        clamped = False
        if dir_cam2[2] >= 0.0:
            # Behind the camera: push to the nearest border in the same direction
            clamped = True
            if s < 1e-6:
                x_ndc, y_ndc = 0.0, 1.0
            else:
                x_ndc /= s
                y_ndc /= s
        else:
            if s > 1.0:
                clamped = True
                x_ndc /= s
                y_ndc /= s
        
        return x_ndc, y_ndc, clamped
    
    def draw_crosshair_overlay(self):
        """Draw a small crosshair showing where camera-1 is pointing in camera-2's view."""
        # Compute NDC in camera-2, then to pixel coords
        x_ndc, y_ndc, clamped = self._compute_crosshair_ndc_in_cam2(self.width)
        px = (x_ndc * 0.5 + 0.5) * self.width
        py = (y_ndc * 0.5 + 0.5) * self.height
        
        # Set up 2D overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Overlay state
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Style: white when normal, yellow when clamped
        if clamped:
            glColor3f(1.0, 0.85, 0.0)
        else:
            glColor3f(1.0, 1.0, 1.0)
        
        glLineWidth(2.0)
        
        # Crosshair lines (~12px total length)
        half_len = 6.0
        glBegin(GL_LINES)
        # Horizontal line
        glVertex2f(px - half_len, py)
        glVertex2f(px + half_len, py)
        # Vertical line
        glVertex2f(px, py - half_len)
        glVertex2f(px, py + half_len)
        glEnd()
        
        # When clamped, draw a small ring to signify clamping
        if clamped:
            radius = 6.0
            segments = 20
            glBegin(GL_LINE_LOOP)
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                vx = px + radius * math.cos(angle)
                vy = py + radius * math.sin(angle)
                glVertex2f(vx, vy)
            glEnd()
        
        # Restore state
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
    
    def save_state(self):
        """Save physics and camera state to file (for window 1)"""
        if self.window_mode == 1:
            try:
                state = {
                    'position': self.physics.position,
                    'velocity': self.physics.velocity,
                    'orientation': self.physics.orientation,
                    'angular_velocity': self.physics.angular_velocity,
                    'camera2_rotation': self.camera2_rotation,
                    'world_vertices': self.world_vertices,
                    'world_faces': self.world_faces,
                    'hoops': self.hoops
                }
                np.save(self.state_file, state, allow_pickle=True)
            except Exception as e:
                pass  # Silently fail if can't save
    
    def load_state(self):
        """Load physics and camera state from file (for window 2)"""
        if self.window_mode == 2:
            try:
                import os
                if os.path.exists(self.state_file):
                    state = np.load(self.state_file, allow_pickle=True).item()
                    self.physics.position = state['position']
                    self.physics.velocity = state['velocity']
                    self.physics.orientation = state['orientation']
                    self.physics.angular_velocity = state['angular_velocity']
                    # Load camera2_rotation from Window 1's WASD input
                    self.camera2_rotation = state['camera2_rotation']
                    
                    # Load world data if not already loaded
                    if not self.world_vertices:
                        self.world_vertices = state.get('world_vertices', [])
                        self.world_faces = state.get('world_faces', [])
                        self.hoops = state.get('hoops', [])
                        self._ground_faces = None
                        self._block_faces = None
            except Exception as e:
                pass  # Silently fail if can't load
    
    def render(self):
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Full window viewport
        glViewport(0, 0, self.width, self.height)
        aspect_ratio = self.width / self.height
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70, aspect_ratio, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Render based on window mode
        if self.window_mode == 1:
            # Window 1: Controller view (follows drone orientation)
            self.update_camera()
        else:
            # Window 2: 360° view (WASD-controlled)
            self.update_camera_secondary()
        
        # Draw world and hoops
        self.draw_world()
        self.draw_hoops()
        
        # Draw crosshair overlay in Window 2 (shows where Window 1 is looking)
        if self.window_mode == 2:
            self.draw_crosshair_overlay()
        
        # Swap buffers
        pygame.display.flip()
    
    def run(self):
        running = True
        print("=" * 60)
        if self.window_mode == 1:
            print("Window 1 - Controller View (FOCUS THIS WINDOW)")
            print("=" * 60)
            print("Controls:")
            print("  DJI Controller:")
            print("    Axis 2 (Channel 2): Throttle (0% to 100%)")
            print("    Axis 1 (Channel 1): Pitch (forward/back tilt)")
            print("    Axis 0 (Channel 0): Roll (left/right tilt)")
            print("    Axis 3 (Channel 3): Yaw (rotation)")
            print("  Keyboard (controls Window 2's 360° camera):")
            print("    W/S: Look down/up in Window 2")
            print("    A/D: Look left/right in Window 2")
            print("    R: Reset drone")
            print("    ESC: Exit")
        else:
            print("Window 2 - 360° View (Read-Only Monitor)")
            print("=" * 60)
            print("This window displays the 360° view controlled by")
            print("WASD keys in Window 1. Keep Window 1 focused.")
            print("  Keyboard:")
            print("    ESC: Exit")
        print("=" * 60)
        
        while running:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.window_mode == 1:
                        self.reset_drone()
            
            # Window 1: Control drone with controller AND camera2 with WASD
            if self.window_mode == 1:
                # Get raw axis values directly
                # Axis 0 = roll, Axis 1 = pitch, Axis 2 = throttle, Axis 3 = yaw
                axis0_raw = self.controller.joystick.get_axis(0) if self.controller.connected and self.controller.joystick else 0.0
                axis1_raw = self.controller.joystick.get_axis(1) if self.controller.connected and self.controller.joystick else 0.0
                axis2_raw = self.controller.joystick.get_axis(2) if self.controller.connected and self.controller.joystick else 0.0
                axis3_raw = self.controller.joystick.get_axis(3) if self.controller.connected and self.controller.joystick else 0.0
                
                # Throttle: Axis 2 from -1 (bottom) to +1 (top) -> map to 0.0 (bottom) to 1.0 (top)
                throttle = (axis2_raw + 1.0) / 2.0
                if throttle < 0.01:
                    throttle = 0.0
                
                # Pitch: Axis 1 (inverted because pygame inverts Y)
                pitch_raw = -axis1_raw
                
                # Roll: Axis 0 (left/right stick movement)
                roll_raw = axis0_raw
                
                # Apply exponential curves to pitch and roll for smoother center, more sensitive edges
                expo_rate = 2.0  # 0.0 = linear, set to 2.0 for 3x sensitivity at max input
                # Formula: output = input * (1 + expo_rate * input^2)
                # At max input (1.0): output = 1.0 * (1 + 2.0) = 3.0 (3x more sensitive)
                pitch = pitch_raw * (1.0 + expo_rate * pitch_raw * pitch_raw)
                roll = roll_raw * (1.0 + expo_rate * roll_raw * roll_raw)
                
                # Yaw: Axis 3 (rotation)
                yaw = axis3_raw
                
                # Set motor speeds (throttle, pitch, roll, and yaw)
                self.physics.set_motor_speeds(throttle, pitch, roll, yaw)
                
                # Update physics
                self.physics.update(self.dt)
                
                # Update camera2 rotation based on keyboard input (WASD) - for Window 2 to display
                keys = pygame.key.get_pressed()
                self.update_camera2_rotation(keys)
                
                # Save state for window 2
                self.save_state()
            
            # Window 2: Load state from Window 1 (read-only display)
            else:
                # Load state from window 1 (includes drone physics AND camera2 rotation)
                self.load_state()
            
            # Render
            self.render()
            
            self.dt = self.clock.tick(60) / 1000.0
        
        pygame.quit()
