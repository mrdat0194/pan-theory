import os
import subprocess

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
inputdir = desktop_path
outdir = desktop_path

for filename in os.listdir(inputdir):
    actual_filename = filename[:-4]
    if (filename.endswith(".mp4") or filename.endswith(".MP4") ):
        subprocess.run(['ffmpeg', '-i', os.path.join(inputdir, filename), '-acodec', 'pcm_s16le', '-ar', '16000', os.path.join(outdir, f'{actual_filename}.wav')])
    else :
        print(f'Skipping {filename}')