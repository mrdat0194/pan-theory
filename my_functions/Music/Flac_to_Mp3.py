import os
import subprocess
from pathlib import Path

def convert_flac_to_mp3(directory):
    """
    Converts all .flac files in the specified directory to .mp3 using ffmpeg.
    """
    path = Path(directory)
    
    if not path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    flac_files = list(path.glob("*.flac"))
    
    if not flac_files:
        print(f"No .flac files found in {directory}")
        return

    print(f"Found {len(flac_files)} .flac files. Starting conversion...")

    for flac_file in flac_files:
        mp3_file = flac_file.with_suffix(".mp3")
        
        print(f"Converting: {flac_file.name} -> {mp3_file.name}")
        
        try:
            # -i: input file
            # -ab: audio bitrate (192k is a good standard)
            # -map_metadata 0: copy metadata from input to output
            # -y: overwrite output if it exists
            subprocess.run([
                'ffmpeg', 
                '-i', str(flac_file), 
                '-ab', '192k', 
                '-map_metadata', '0', 
                '-y', 
                str(mp3_file)
            ], check=True, capture_output=True)
            print(f"Successfully converted {flac_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {flac_file.name}")
            print(f"Error: {e.stderr.decode()}")
        except FileNotFoundError:
            print("Error: 'ffmpeg' not found in system PATH. Please ensure ffmpeg is installed.")
            break

if __name__ == "__main__":
    target_dir = r"C:\Users\mrdat\PycharmProjects\pan-theory\my_functions\Music"
    convert_flac_to_mp3(target_dir)
