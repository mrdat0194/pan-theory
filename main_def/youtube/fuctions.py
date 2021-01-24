import youtube_dl
import time
from youtube_dl.utils import DownloadError
from main_def.crud.sql.datasource import get_one_youtube_url_and_youtube_uploader_by_youtube_url
from main_def import youtube_com_cookies_path
import json
from numpy import random
import traceback


def get_raw_title_uploader_from_youtube_url(youtube_url: str):
    ytdl_options = {
        # "cachedir": False,
        "quiet": True,
        "nocheckcertificate": True,
        "restrictfilenames": True,
        "cookiefile": youtube_com_cookies_path,
    }

    try:
        ydl = youtube_dl.YoutubeDL(ytdl_options)

        result = ydl.extract_info(
            youtube_url,
            download=False  # We just want to extract the info
        )
        youtube_info_result = {"youtube_url": youtube_url, "uploader": result.get('uploader'),
                               "youtube_title": result.get('title')}
    except DownloadError as ex:
        youtube_info_result = {"youtube_url": youtube_url, "uploader": f"{ex}", "youtube_title": f"{ex}"}
    except:  # noqa
        youtube_info_result = {"youtube_url": youtube_url, "uploader": "Error: Unknown error",
                               "youtube_title": "Error: Unknown error"}
    x = random.uniform(0.5, 3)
    time.sleep(x)
    return youtube_info_result


def get_youtube_title_and_youtube_uploader_from_youtube_url(youtube_url: str):
    db_datasources = get_one_youtube_url_and_youtube_uploader_by_youtube_url(youtube_url)
    if not db_datasources:
        youtube_info_result = get_raw_title_uploader_from_youtube_url(youtube_url)
    else:
        for db_datasource in db_datasources:
            info = db_datasource.info.get('source', None)
            youtube_title = info.get('title', None)
            uploader = info.get('uploader', None)
            youtube_info_result = {"youtube_url": youtube_url, "uploader": uploader,
                                   "youtube_title": youtube_title}
    return youtube_info_result


if __name__ == "__main__":
    start_time = time.time()
    youtube_urls = [
        "https://www.youtube.com/watch?v=OQpWoPi3H28",
        "https://www.youtube.com/watch?v=uWM9vJad7hY",
        "https://www.youtube.com/watch?v=UkCqIpWoRcY",
        "https://www.youtube.com/watch?v=nCvXtx9QPd8",
        "https://www.youtube.com/watch?v=HyZnq44kVBQ",
        "https://www.youtube.com/watch?v=psZ1g9fMfeo",
        "https://www.youtube.com/watch?v=xIEhVH7wQw4",
        "https://www.youtube.com/watch?v=NPUgrtEWs0Y"


    ]
    for youtube_url in youtube_urls:
        print(youtube_url)
        get_youtube_title_and_youtube_uploader_from_youtube_url(youtube_url)
        # get_raw_title_uploader_from_youtube_url(youtube_url)

    t2 = time.time() - start_time
    print(t2)

