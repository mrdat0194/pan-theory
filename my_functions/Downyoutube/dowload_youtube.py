import youtube_dl
import os
import argparse

def download_youtube():
    parser = argparse.ArgumentParser(description='NA')
    link = 'https://www.youtube.com/watch?v=M1A0epTU-jc'
    parser.add_argument('-l', '--link', default= link, help='JSON file')
    args = parser.parse_args()
    link = args.link

    ydl_opts = {}
    os.chdir('.')
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

if  __name__ == "__main__":
    download_youtube()
    # find_max_int(path)
    
