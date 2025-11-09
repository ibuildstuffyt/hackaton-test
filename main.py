"""
Main entry point for FPV Simulator with Marble World integration
"""
import sys
import os
from marble_api import MarbleWorldAPI
from fpv_simulator import FPVSimulator
import json


def load_world_from_file(file_path: str):
    """
    Load a simple world mesh from a file
    For now, creates a simple test world
    """
    # This is a placeholder - in a real implementation, you'd parse GLTF/OBJ files
    # For now, we'll create a simple test environment
    vertices = []
    faces = []
    
    # Create a simple test environment (ground + various obstacles)
    # Ground - expanded
    ground_size = 100.0
    vertices.extend([
        [-ground_size, 0, -ground_size],
        [ground_size, 0, -ground_size],
        [ground_size, 0, ground_size],
        [-ground_size, 0, ground_size]
    ])
    faces.append([0, 1, 2])
    faces.append([0, 2, 3])
    
    import math
    
    # Helper function to add a cube
    def add_cube(vertices, faces, x, z, size, height):
        base_idx = len(vertices)
        vertices.extend([
            [x - size, 0, z - size],
            [x + size, 0, z - size],
            [x + size, 0, z + size],
            [x - size, 0, z + size],
            [x - size, height, z - size],
            [x + size, height, z - size],
            [x + size, height, z + size],
            [x - size, height, z + size]
        ])
        # Bottom
        faces.append([base_idx, base_idx + 1, base_idx + 2])
        faces.append([base_idx, base_idx + 2, base_idx + 3])
        # Top
        faces.append([base_idx + 4, base_idx + 5, base_idx + 6])
        faces.append([base_idx + 4, base_idx + 6, base_idx + 7])
        # Sides
        faces.append([base_idx, base_idx + 4, base_idx + 5])
        faces.append([base_idx, base_idx + 5, base_idx + 1])
        faces.append([base_idx + 1, base_idx + 5, base_idx + 6])
        faces.append([base_idx + 1, base_idx + 6, base_idx + 2])
        faces.append([base_idx + 2, base_idx + 6, base_idx + 7])
        faces.append([base_idx + 2, base_idx + 7, base_idx + 3])
        faces.append([base_idx + 3, base_idx + 7, base_idx + 4])
        faces.append([base_idx + 3, base_idx + 4, base_idx])
    
    # Helper function to add a pyramid
    def add_pyramid(vertices, faces, x, z, base_size, height):
        base_idx = len(vertices)
        # Base square
        vertices.extend([
            [x - base_size, 0, z - base_size],
            [x + base_size, 0, z - base_size],
            [x + base_size, 0, z + base_size],
            [x - base_size, 0, z + base_size],
            [x, height, z]  # Top point
        ])
        # Base
        faces.append([base_idx, base_idx + 1, base_idx + 2])
        faces.append([base_idx, base_idx + 2, base_idx + 3])
        # Sides
        faces.append([base_idx, base_idx + 4, base_idx + 1])
        faces.append([base_idx + 1, base_idx + 4, base_idx + 2])
        faces.append([base_idx + 2, base_idx + 4, base_idx + 3])
        faces.append([base_idx + 3, base_idx + 4, base_idx])
    
    # Helper function to add a cylinder (approximated with octagon)
    def add_cylinder(vertices, faces, x, z, radius, height, segments=8):
        base_idx = len(vertices)
        # Bottom circle
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertices.append([x + radius * math.cos(angle), 0, z + radius * math.sin(angle)])
        # Top circle
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertices.append([x + radius * math.cos(angle), height, z + radius * math.sin(angle)])
        # Bottom face
        for i in range(segments - 2):
            faces.append([base_idx, base_idx + i + 1, base_idx + i + 2])
        # Top face
        for i in range(segments - 2):
            faces.append([base_idx + segments, base_idx + segments + i + 2, base_idx + segments + i + 1])
        # Sides
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([base_idx + i, base_idx + next_i, base_idx + segments + next_i])
            faces.append([base_idx + i, base_idx + segments + next_i, base_idx + segments + i])
    
    # Add various obstacles in a grid pattern (reduced for performance)
    obstacle_positions = []
    spacing = 10.0
    grid_size = 6  # Reduced from 10 to 6 for better performance
    
    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            if i == 0 and j == 0:
                continue
            # Skip some positions to reduce object count
            if (i + j) % 2 == 0:
                continue
            x = i * spacing
            z = j * spacing
            obstacle_positions.append((x, z))
    
    # Add different types of obstacles
    import random
    random.seed(42)  # For consistent generation
    
    for x, z in obstacle_positions:
        obj_type = random.choice(['cube', 'pyramid', 'cylinder', 'tall_cube'])
        
        if obj_type == 'cube':
            add_cube(vertices, faces, x, z, 1.5, 2.0)
        elif obj_type == 'pyramid':
            add_pyramid(vertices, faces, x, z, 1.5, 3.0)
        elif obj_type == 'cylinder':
            add_cylinder(vertices, faces, x, z, 1.2, 2.5, segments=6)  # Reduced segments from 8 to 6
        elif obj_type == 'tall_cube':
            add_cube(vertices, faces, x, z, 1.0, 4.0)
    
    return vertices, faces


def main():
    """Main function"""
    print("=" * 60)
    print("FPV Simulator - Marble World Integration")
    print("=" * 60)
    
    # Initialize Marble World API
    api = MarbleWorldAPI()
    
    # Try to list available worlds
    print("\nFetching available worlds from Marble World...")
    worlds = api.list_worlds()
    
    if worlds:
        print(f"\nFound {len(worlds)} available world(s):")
        for i, world in enumerate(worlds):
            print(f"  {i + 1}. {world.get('name', 'Unknown')} (ID: {world.get('id', 'N/A')})")
        
        # For now, use the first world or let user choose
        if len(worlds) > 0:
            selected_world = worlds[0]
            print(f"\nUsing world: {selected_world.get('name', 'Unknown')}")
            
            # Try to export/download the world
            world_id = selected_world.get('id')
            if world_id:
                print(f"Exporting world {world_id}...")
                export_data = api.export_world(world_id, format="gltf")
                if export_data:
                    print("World export initiated. (Note: Full GLTF parsing not yet implemented)")
                    print("Using test environment for now...")
    else:
        print("No worlds found or API connection failed.")
        print("Using test environment...")
    
    # Initialize simulator
    print("\nInitializing FPV Simulator...")
    simulator = FPVSimulator()
    
    # Load world mesh (for now, using test world)
    print("Loading world mesh...")
    vertices, faces = load_world_from_file("test_world")
    simulator.load_world_mesh(vertices, faces)
    
    # Run simulator
    print("\nStarting simulator...")
    print("Make sure your DJI controller is connected via USB!")
    simulator.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



