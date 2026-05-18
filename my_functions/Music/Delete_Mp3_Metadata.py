import os
import subprocess
import sys
import io
from pathlib import Path

# Fix for Unicode encoding issues in Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def remove_metadata_from_mp3s(directory):
    """
    Recursively scans the directory for .mp3 files and removes all metadata
    (including ID3 tags and embedded album art) using FFmpeg without re-encoding.
    """
    root_path = Path(directory)
    if not root_path.exists():
        print(f"Error: Directory '{directory}' does not exist.", flush=True)
        return

    print(f"Scanning for MP3 files in: {directory}", flush=True)
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(".mp3"):
                mp3_files.append(Path(root) / filename)

    # Sort for consistent processing order
    mp3_files.sort(key=lambda x: x.name)
    total_files = len(mp3_files)
    
    if total_files == 0:
        print("No MP3 files found to process.", flush=True)
        return

    print(f"Found {total_files} MP3 files. Starting metadata removal process...", flush=True)
    
    successful_count = 0
    failed_count = 0

    for idx, file_path in enumerate(mp3_files, start=1):
        # Create a temporary file path alongside the original file
        temp_path = file_path.with_name(f"{file_path.stem}_temp_stripped{file_path.suffix}")
        
        print(f"[{idx}/{total_files}] Processing: {file_path.name}", flush=True)
        
        try:
            # Command to copy audio stream lossless and drop all metadata/image streams
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-map', '0:a',
                '-map_metadata', '-1',
                '-codec', 'copy',
                '-y',
                str(temp_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, check=True)
            
            # Verify temporary file was successfully created and has data
            if temp_path.exists() and temp_path.stat().st_size > 0:
                # Replace the original file with the stripped temporary file
                os.replace(temp_path, file_path)
                successful_count += 1
            else:
                print(f"  -> Error: Stripped output file is missing or empty for {file_path.name}", flush=True)
                failed_count += 1
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                    
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode('utf-8', errors='replace').strip()
            # Extract relevant error message summary from ffmpeg stderr output
            err_summary = err_msg.split('\n')[-1] if err_msg else "Unknown FFmpeg error"
            print(f"  -> Failed to strip metadata from {file_path.name}: {err_summary}", flush=True)
            failed_count += 1
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            print(f"  -> Unexpected error processing {file_path.name}: {e}", flush=True)
            failed_count += 1
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    print("\n--- Metadata Removal Summary ---", flush=True)
    print(f"Total files processed: {total_files}", flush=True)
    print(f"Successfully stripped: {successful_count}", flush=True)
    print(f"Failed to strip:       {failed_count}", flush=True)

if __name__ == "__main__":
    target_directory = r"C:\Users\mrdat\Desktop\Mp3 files"
    remove_metadata_from_mp3s(target_directory)
    print("\nAll tasks completed successfully.", flush=True)
