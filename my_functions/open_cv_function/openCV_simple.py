import cv2
import time
from my_functions.timer import timer

@timer
def decode_fourcc(v: int):
    '''
    avc1: H.264
    av01: av1
    https://github.com/opencv/opencv/blob/master/samples/python/video_v4l2.py
    '''
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

@timer
def get_video_decode(url: str):
    vidcap = cv2.VideoCapture(url)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 3000)  # just cue to 20 sec. position
    fourcc = vidcap.get(cv2.CAP_PROP_FOURCC)
    decode_name = decode_fourcc(fourcc)
    return decode_name

@timer
def get_video_image(url: str):
    vidcap = cv2.VideoCapture(url)
    success,image = vidcap.read()
    if success:
        cv2.imwrite("frame3sec.jpg", image)     # save frame as JPEG file
        cv2.imshow('joy',image)
        # cv2.waitKey()


if __name__ == "__main__":
    start_time = time.time()
    urls = [
       "https://berserker4.samV-vid.com/samV-us/videos/video_42AED2696FCD452E8D358B0CDBE28B6F.mp4"
    ]
    for url in urls:
        k = get_video_decode(url)
        print(f"{k} ----{url}")
        get_video_image(url)

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
# vidcap.get(CV_CAP_PROP_FOURCC);
