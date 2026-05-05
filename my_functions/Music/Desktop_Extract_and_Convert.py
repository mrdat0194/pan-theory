import os
import subprocess
import zipfile
import sys
import io
from pathlib import Path

# Fix for Unicode encoding issues in Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def unzip_and_delete(directory):
    """
    Unzips all .zip files in the directory and deletes them after extraction.
    Prioritizes 'Khanh Ly' files.
    """
    path = Path(directory)
    zip_files = list(path.glob("**/*.zip"))
    
    if not zip_files:
        print("No .zip files found.")
        return

    # Prioritize 'Khanh Ly' by sorting (reverse sort puts 'K' higher, or custom sort)
    zip_files.sort(key=lambda x: (0 if "Khanh Ly" in x.name else 1, x.name))

    print(f"Found {len(zip_files)} zip files. Starting extraction (Prioritizing Khanh Ly)...", flush=True)
    
    for zip_path in zip_files:
        print(f"Extracting: {zip_path.name}", flush=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_path.parent)
            
            print(f"Successfully extracted {zip_path.name}. Deleting zip file...", flush=True)
            zip_path.unlink()
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}", flush=True)

def convert_flac_to_mp3_recursive(src_dir, dest_dir):
    """
    Recursively converts all .flac files from src_dir to .mp3 in dest_dir.
    Prioritizes 'Khanh Ly' files.
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    if not src_path.exists():
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    print("Gathering FLAC files for conversion...", flush=True)
    all_flac_files = []
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.lower().endswith(".flac"):
                all_flac_files.append(Path(root) / filename)

    # Prioritize 'Khanh Ly'
    all_flac_files.sort(key=lambda x: (0 if "Khanh Ly" in x.as_posix() else 1, x.as_posix()))

    print(f"Starting conversion for {len(all_flac_files)} files (Prioritizing Khanh Ly)...", flush=True)
    
    for flac_file in all_flac_files:
        rel_path = flac_file.relative_to(src_path)
        mp3_file = dest_path / rel_path.with_suffix(".mp3")
        
        if mp3_file.exists():
            # If MP3 already exists, we can delete the FLAC now to save space
            print(f"Already converted: {mp3_file.name}. Deleting original FLAC.", flush=True)
            try:
                flac_file.unlink()
            except Exception as e:
                print(f"Could not delete {flac_file.name}: {e}", flush=True)
            continue

        mp3_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting: {rel_path} -> {mp3_file.name}", flush=True)
        
        try:
            subprocess.run([
                'ffmpeg', 
                '-i', str(flac_file), 
                '-ab', '192k', 
                '-map_metadata', '0', 
                '-y', 
                str(mp3_file)
            ], check=True, capture_output=True)
            
            # Delete the flac file after successful conversion to save space
            print(f"Successfully converted. Deleting original: {flac_file.name}", flush=True)
            flac_file.unlink()
            
        except subprocess.CalledProcessError as e:
            # Safely handle error output decoding
            err_msg = e.stderr.decode('utf-8', errors='replace')
            print(f"Failed to convert {flac_file.name}: {err_msg}", flush=True)
        except Exception as e:
            print(f"Unexpected error converting {flac_file.name}: {e}", flush=True)

if __name__ == "__main__":
    src = r"C:\Users\mrdat\Desktop\flac"
    dest = r"C:\Users\mrdat\Desktop\Mp3"
    
    unzip_and_delete(src)
    convert_flac_to_mp3_recursive(src, dest)
    print("\nAll tasks completed successfully.", flush=True)
