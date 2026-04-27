'''
Chet Gnegy
www.chetgnegy.com
Free Photo Collage Generator
GPL License

'''
# C:\Users\mrdat\Downloads\collage.py

from PIL import Image, ImageEnhance
import PIL.ExifTags as ExifTags
import glob, os
import operator
import random
import sys, getopt

bestPicks = {}

def topN(arr, N):
    sorted_arr = sorted(arr.items(), key=operator.itemgetter(1))
    return sorted_arr[0:N]

# Returns the elementwise average color
def getAvgColor(im):
    R_sum = 0
    G_sum = 0
    B_sum = 0
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            (R, G, B) = im.getpixel((i, j))
            R_sum += R
            G_sum += G
            B_sum += B
    total = im.size[0] * im.size[1]
    R_sum /= total
    G_sum /= total
    B_sum /= total
    return (R_sum, G_sum, B_sum)

def quantize(trip, n):
    R = (int(trip[0]) >> n) << n
    G = (int(trip[1]) >> n) << n
    B = (int(trip[2]) >> n) << n
    return (R, G, B)

# Gets the most frequently used color in the image
def getFrequent(im):
    cols = {}
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pix = im.getpixel((i, j))
            pix = quantize(pix, 3)
            cols[pix] = cols.get(pix, 0) + 1
    freq = max(cols.items(), key=operator.itemgetter(1))[0]
    return freq

# A metric for determining match quality
def matchQuality(match, target):
    match = quantize(match, 3)
    R = match[0]
    G = match[1]
    B = match[2]
    dR = target[0] - R
    dG = target[1] - G
    dB = target[2] - B
    return abs(dR) + abs(dG) + abs(dB)

def main(argv):
    # Handle input arguments
    thumb_size = 200
    target_tiles = 200
    blend_amt = 0.7
    outputfile = 'outfile.jpeg'
    directory = '/'  # Default directory is current directory
    # New: Add brightness and contrast factors as optional arguments
    brightness_factor = 1.1
    contrast_factor = 1.1
    try:
        opts, args = getopt.getopt(argv, "hi:o:d:t:n:b:B:C:", ["ifile=", "ofile=", "dir", "tilesize=", "numtiles", "blend_amt", "brightness=", "contrast="])
    except getopt.GetoptError:
        print('collage.py -i <inputfile> -d <directory> -o <outputfile> -t <tilesize> -n <numtiles> -b <blend_amt> -B <brightness> -C <contrast>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('collage.py -i <inputfile> -d <directory> -o <outputfile> -t <tilesize> -n <numtiles> -b <blend_amt> -B <brightness> -C <contrast>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-d", "--dir"):
            directory = arg
        elif opt in ("-t", "--tilesize"):
            thumb_size = int(arg)
        elif opt in ("-n", "--numtiles"):
            target_tiles = int(arg)
        elif opt in ("-b", "--blend_amt"):
            blend_amt = float(arg)
        elif opt in ("-B", "--brightness"):
            brightness_factor = float(arg)
        elif opt in ("-C", "--contrast"):
            contrast_factor = float(arg)

    input_dir = os.path.dirname(os.path.abspath(inputfile))
    input_dir_save = os.path.dirname(os.path.abspath(inputfile))
    directory = os.path.join(input_dir, "source")

    # Creates thumbnails for each image in the source set
    if not os.path.exists("source/"):
        os.makedirs("source/")

    print("Making thumbnails...")
    for infile in glob.glob(directory + "/*"):
        try:
            file, ext = os.path.splitext(infile)

            # Strip off the directory
            if "/" in file:
                file = file[file.rfind("/") + 1:]

            # We don't wanna make anything that is already there
            if os.path.isfile("source/" + file + "_thumb.jpg") or "_thumb" in file:
                continue

            # Only opens image files
            try:
                im = Image.open(infile)
            except:
                continue

            # Rotates the thumbnail to the correct orientation
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(im._getexif().items())
                if exif[orientation] == 3:
                    im = im.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    im = im.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    im = im.rotate(90, expand=True)
            except:
                pass

            # Square image from the center
            min_dimension = min(im.size)
            if min_dimension == im.size[0]:
                x = 0
                y = im.size[1] / 2 - min_dimension / 2
            else:
                x = im.size[0] / 2 - min_dimension / 2
                y = 0
            im = im.crop((x, y, x + min_dimension, y + min_dimension))

            im.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            
            path_save_img = os.path.join(input_dir_save, "sources")
            path_save_img = os.path.join(path_save_img,  file + "_thumb.jpg")
          
            im.save(path_save_img, "JPEG")

        except Exception as err:
            print("Error reading file: ", file, ext, "-", err)

    thumbs_lib = {}
    thumbs_lib_avg = {}
    thumbs_lib_freq = {}
    found_some = False
    # Collects data on the individual images
    print("Collecting data...")
    for infile in glob.glob("source/*_thumb.jpg"):
        file, ext = os.path.splitext(infile)
        try:
            im = Image.open(infile)
            thumbs_lib[file] = im
            thumbs_lib_avg[file] = getAvgColor(im)
            thumbs_lib_freq[file] = getFrequent(im)
            found_some = True
        except Exception as err:
            print("Error reading file: ", file, ext, "-", err)

    if not found_some:
        print("There are no thumbnails in this directory!\n")
        return

    print("Resizing target...")
    # Makes a small copy of the target file (with filtering)
    target = Image.open(inputfile)

    # --- Increase brightness and/or contrast of the original file before collage ---
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(target)
        target = enhancer.enhance(brightness_factor)
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(target)
        target = enhancer.enhance(contrast_factor)
    # ------------------------------------------------------------------------------

    [x, y] = target.size
    x_ = x / (1.0 * min(x, y))
    y_ = y / (1.0 * min(x, y))
    x_tiles = int(x_ * target_tiles)
    y_tiles = int(y_ * target_tiles)
    target_thumb = target.copy()
    target_thumb.thumbnail((x_tiles, y_tiles), Image.LANCZOS)

    last_choose = ""
    # Creates a blank image that needs to be filled by the thumbnails
    final = Image.new("RGB", (thumb_size * target_thumb.size[0], thumb_size * target_thumb.size[1]), "black")
    print("Making collage...")

    # For each tile in the target, find the best matching thumbnail and paste it
    for j in range(target_thumb.size[1]):
        for i in range(target_thumb.size[0]):
            target_pix = target_thumb.getpixel((i, j))
            bestPicks = {}
            for key in thumbs_lib_avg:
                metric1 = matchQuality(thumbs_lib_freq[key], target_pix)
                metric2 = matchQuality(thumbs_lib_avg[key], target_pix)
                metric = min(metric1, metric2) + (metric1 + metric2) / 2
                bestPicks[key] = metric

            N = min(100, len(bestPicks))
            if N == 0:
                continue  # No thumbnails to choose from

            choose = topN(bestPicks, N)
            k = random.randint(0, N - 1)

            # Avoid repeating the last chosen thumbnail
            attempts = 0
            while N > 1 and last_choose == choose[k][0] and attempts < 10:
                k = random.randint(0, N - 1)
                attempts += 1

            last_choose = choose[k][0]

            # Ensure the thumbnail is exactly the right size before pasting
            thumb_img = thumbs_lib[last_choose]
            if thumb_img.size != (thumb_size, thumb_size):
                thumb_img = thumb_img.resize((thumb_size, thumb_size), Image.LANCZOS)

            # Paste the thumbnail at the correct position
            final.paste(
                thumb_img,
                (
                    i * thumb_size,
                    j * thumb_size
                )
            )

    print("Saving...")

    # --- WHY DOES THE RESULT LOOK DULL? ---
    # The result looks dull because of the blending step below.
    # The line below blends the collage with the original image using the blend_amt parameter.
    # If blend_amt is high (e.g., 0.5), the original image is strongly blended over the collage,
    # which reduces the contrast and vibrancy of the collage tiles, making the result look "dull" or "washed out".
    # To make the result less dull, reduce blend_amt (e.g., 0.2 or 0), or skip blending entirely.
    # --------------------------------------

    # Resize the target image to match the final collage size for blending
    bigger = target.resize(final.size, Image.LANCZOS)
    final = Image.blend(final, bigger, blend_amt)
    final.save(outputfile, "JPEG")

if __name__ == "__main__":
    main(sys.argv[1:])