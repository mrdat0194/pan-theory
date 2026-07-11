import os
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def is_safe_filename(filename: str) -> bool:
    """
    Check if the filename is safe to use.
    Rejects filenames starting with '-' to prevent argument injection
    if they were used without proper prefixing.
    Also rejects filenames that look like protocols (e.g., 'file:').
    """
    if filename.startswith('-'):
        return False
    if ':' in filename:
        # Many ffmpeg protocols use ':' (e.g., http:, concat:, subfile:)
        return False
    return True

def convert_mp4_to_wav(input_dir: Path, output_dir: Path):
    """
    Converts all .mp4 files in input_dir to .wav in output_dir.
    """
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not (filename.lower().endswith(".mp4")):
            continue

        if not is_safe_filename(filename):
            logger.warning(f"Skipping potentially unsafe filename: {filename}")
            continue

        actual_filename = Path(filename).stem
        input_file = (input_dir / filename).resolve()
        output_file = (output_dir / f"{actual_filename}.wav").resolve()

        logger.info(f"Converting {filename} to {output_file.name}")

        try:
            # Using list for arguments and ensuring absolute paths for input/output
            # mitigates most command injection and argument injection risks.
            subprocess.run([
                'ffmpeg',
                '-i', str(input_file),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-y',  # Overwrite output if exists
                str(output_file)
            ], check=True, capture_output=True, text=True)
            logger.info(f"Successfully converted {filename}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {filename}: {e.stderr}")
        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred while converting {filename}: {e}")

def main():
    desktop_path = Path.home() / "Desktop"
    inputdir = desktop_path
    outdir = desktop_path

    convert_mp4_to_wav(inputdir, outdir)

if __name__ == "__main__":
    main()
