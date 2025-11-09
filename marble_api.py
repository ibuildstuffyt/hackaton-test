"""
Marble World API Client
Handles importing worlds from marble.worldlabs.ai
"""
import requests
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class MarbleWorldAPI:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.marble.worldlabs.ai"):
        """
        Initialize the Marble World API client
        
        Args:
            api_key: API key for authentication (can also be set via MARBLE_API_KEY env var)
            base_url: Base URL for the API
        """
        self.api_key = api_key or os.getenv("MARBLE_API_KEY")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
    
    def list_worlds(self) -> List[Dict]:
        """
        List available worlds from Marble World
        
        Returns:
            List of world dictionaries with metadata
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/worlds")
            response.raise_for_status()
            return response.json().get("worlds", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching worlds: {e}")
            return []
    
    def get_world(self, world_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific world
        
        Args:
            world_id: ID of the world to retrieve
            
        Returns:
            World data dictionary or None if not found
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/worlds/{world_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching world {world_id}: {e}")
            return None
    
    def export_world(self, world_id: str, format: str = "gltf") -> Optional[Dict]:
        """
        Export a world in a specific format for use in the simulator
        
        Args:
            world_id: ID of the world to export
            format: Export format (gltf, obj, fbx, etc.)
            
        Returns:
            Export data dictionary with download URL or file data
        """
        try:
            response = self.session.post(
                f"{self.base_url}/v1/worlds/{world_id}/export",
                json={"format": format}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error exporting world {world_id}: {e}")
            return None
    
    def download_world_assets(self, world_id: str, output_dir: str = "worlds") -> Optional[str]:
        """
        Download world assets to local directory
        
        Args:
            world_id: ID of the world to download
            output_dir: Directory to save assets
            
        Returns:
            Path to downloaded world directory or None if failed
        """
        export_data = self.export_world(world_id, format="gltf")
        if not export_data:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        world_dir = os.path.join(output_dir, world_id)
        os.makedirs(world_dir, exist_ok=True)
        
        # Download the main model file
        download_url = export_data.get("download_url")
        if download_url:
            response = requests.get(download_url)
            response.raise_for_status()
            
            file_path = os.path.join(world_dir, f"{world_id}.gltf")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded world to {file_path}")
            return world_dir
        
        return None




