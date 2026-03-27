import os
import subprocess

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
inputdir = desktop_path
outdir = desktop_path

for filename in os.listdir(inputdir):
    actual_filename = filename[:-4]
    if (filename.endswith(".avi") or filename.endswith(".AVI")):
        subprocess.run(['ffmpeg', '-i', os.path.join(inputdir, filename), '-c:v', 'libx264', '-crf', '19', '-preset', 'slow', '-c:a', 'aac', '-b:a', '192k', '-ac', '2', os.path.join(outdir, f'{actual_filename}.mp4')])
    else :
        print(f'Skipping {filename}')
