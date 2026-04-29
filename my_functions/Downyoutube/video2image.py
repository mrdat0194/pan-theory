import argparse
import cv2
import os 
print(cv2.__version__)

def extractImages(pathIn, pathOut, name):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        try:
            cv2.imwrite("{}/{}_frame{}.jpg".format(pathOut,name,count), image)     # save frame as JPEG file
            count = count + 1
        except Exception:
            print("=== {} --- {}".format(name,count))
            
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    pathIn = args.pathIn
    pathOut = args.pathOut

    if os.path.exists(pathOut) is False:
        os.mkdir(pathOut)

    for video in os.listdir(pathIn):
        path_video = os.path.join(pathIn,video)
        name = video.split(".")[0]
        extractImages(path_video, pathOut, name)
