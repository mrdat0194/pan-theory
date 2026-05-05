import os
import subprocess
from pathlib import Path

def convert_flac_to_mp3_recursive(src_dir, dest_dir):
    """
    Recursively converts all .flac files from src_dir to .mp3 in dest_dir,
    preserving the directory structure.
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    if not src_path.exists():
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Walk through the directory tree
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.lower().endswith(".flac"):
                flac_file = Path(root) / filename
                
                # Calculate the relative path from the source directory
                rel_path = flac_file.relative_to(src_path)
                
                # Create the corresponding path in the destination directory
                mp3_file = dest_path / rel_path.with_suffix(".mp3")
                
                # Ensure the destination subdirectory exists
                mp3_file.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"Converting: {rel_path} -> {mp3_file.name}", flush=True)
                
                try:
                    # ffmpeg command
                    subprocess.run([
                        'ffmpeg', 
                        '-i', str(flac_file), 
                        '-ab', '192k', 
                        '-map_metadata', '0', 
                        '-y', 
                        str(mp3_file)
                    ], check=True, capture_output=True)
                    print(f"Successfully converted: {mp3_file.name}", flush=True)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to convert {filename}")
                    print(f"Error: {e.stderr.decode()}")
                except FileNotFoundError:
                    print("Error: 'ffmpeg' not found. Please install ffmpeg and add it to your PATH.")
                    return

if __name__ == "__main__":
    source = r"C:\Users\mrdat\Desktop\flac"
    destination = r"C:\Users\mrdat\Desktop\Mp3"
    
    convert_flac_to_mp3_recursive(source, destination)
    print("\nConversion process complete.")
