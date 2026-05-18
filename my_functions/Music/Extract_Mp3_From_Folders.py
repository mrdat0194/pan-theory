import os
import shutil
import sys
import io
from pathlib import Path

# Fix for Unicode encoding issues in Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_mp3_and_cleanup(directory):
    """
    Recursively finds all .mp3 files in subfolders of the given directory,
    moves them to the root of the directory, and then deletes all subfolders.
    """
    root_path = Path(directory)
    
    if not root_path.exists():
        print(f"Error: Directory '{directory}' does not exist.", flush=True)
        return

    print(f"Gathering MP3 files from subfolders of: {directory}", flush=True)
    all_mp3_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(".mp3"):
                file_path = Path(root) / filename
                # Only collect files that are inside subfolders, not the root folder itself
                if file_path.parent != root_path:
                    all_mp3_files.append(file_path)

    # Sort files by name for consistent processing
    all_mp3_files.sort(key=lambda x: x.name)
    print(f"Found {len(all_mp3_files)} MP3 files in subfolders to move.", flush=True)
    
    moved_count = 0
    for file_path in all_mp3_files:
        dest_path = root_path / file_path.name
        
        # Handle filename collisions safely to prevent overwriting files
        if dest_path.exists():
            base_name = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while True:
                new_dest_path = root_path / f"{base_name}_{counter}{suffix}"
                if not new_dest_path.exists():
                    dest_path = new_dest_path
                    break

        print(f"Moving: {file_path.relative_to(root_path)} -> {dest_path.name}", flush=True)
        try:
            shutil.move(str(file_path), str(dest_path))
            moved_count += 1
        except Exception as e:
            print(f"Error moving {file_path.name}: {e}", flush=True)

    print(f"Successfully moved {moved_count} MP3 files to the root directory.", flush=True)

    # Delete all subfolders inside the root directory
    print(f"Cleaning up folders in {directory}...", flush=True)
    for item in root_path.iterdir():
        if item.is_dir():
            try:
                shutil.rmtree(item)
                print(f"Deleted folder: {item.name}", flush=True)
            except Exception as e:
                print(f"Could not delete folder {item.name}: {e}", flush=True)

if __name__ == "__main__":
    target_directory = r"C:\Users\mrdat\Desktop\Mp3 files"
    extract_mp3_and_cleanup(target_directory)
    print("\nAll tasks completed successfully.", flush=True)
