'''
Chet Gnegy
www.chetgnegy.com
Free Photo Collage Generator
GPL License
'''

from PIL import Image, ImageEnhance, ImageOps
import glob
import os
import operator
import random
import sys
import argparse

# Backwards compatibility check for PIL Resampling filter constants
if hasattr(Image, "Resampling"):
    LANCZOS = Image.Resampling.LANCZOS
else:
    LANCZOS = Image.LANCZOS

def getAvgColor(im):
    """
    Returns the average color of an image using Pillow's fast resampling.
    """
    # Resize the image to 1x1 to automatically and efficiently compute average color.
    one_pixel = im.resize((1, 1), LANCZOS)
    return one_pixel.getpixel((0, 0))

def quantize(trip, n):
    R = (int(trip[0]) >> n) << n
    G = (int(trip[1]) >> n) << n
    B = (int(trip[2]) >> n) << n
    return (R, G, B)

def getFrequent(im):
    """
    Gets the most frequently used color in the image using Pillow's getcolors.
    """
    # Downscale for performance
    small = im.resize((32, 32))
    cols = {}
    for pixel in small.getdata():
        q_pixel = quantize(pixel[:3], 3)
        cols[q_pixel] = cols.get(q_pixel, 0) + 1
    if not cols:
        return (0, 0, 0)
    freq = max(cols.items(), key=operator.itemgetter(1))[0]
    return freq

def matchQuality(match, target):
    match = quantize(match, 3)
    R = match[0]
    G = match[1]
    B = match[2]
    dR = target[0] - R
    dG = target[1] - G
    dB = target[2] - B
    return abs(dR) + abs(dG) + abs(dB)

def topN(arr, N):
    sorted_arr = sorted(arr.items(), key=operator.itemgetter(1))
    return sorted_arr[0:N]

def main():
    parser = argparse.ArgumentParser(
        description="Free Photo Collage Generator. Refactored for speed, cross-platform safety, and memory controls."
    )
    parser.add_argument("-i", "--ifile", required=True, help="Path to the target input image.")
    parser.add_argument("-d", "--dir", default=None, help="Directory containing source images to make collage tiles. Defaults to a 'source' folder next to the input file.")
    parser.add_argument("-o", "--ofile", default="outfile.jpeg", help="Output collage filename.")
    parser.add_argument("-t", "--tilesize", type=int, default=200, help="Thumbnail tile size in pixels (width/height).")
    parser.add_argument("-n", "--numtiles", type=int, default=200, help="Target tile count along the minimum dimension.")
    parser.add_argument("-b", "--blend-amt", type=float, default=0.2, help="Amount to blend original image over collage [0.0 to 1.0]. Lower values make collage more vibrant.")
    parser.add_argument("-B", "--brightness", type=float, default=1.1, help="Brightness enhancement factor for the target image.")
    parser.add_argument("-C", "--contrast", type=float, default=1.1, help="Contrast enhancement factor for the target image.")
    parser.add_argument("-m", "--max-dim", type=int, default=8000, help="Maximum allowed width/height of the final collage in pixels to prevent memory failure.")
    
    args = parser.parse_args()

    inputfile = args.ifile
    outputfile = args.ofile
    thumb_size = args.tilesize
    target_tiles = args.numtiles
    blend_amt = args.blend_amt
    brightness_factor = args.brightness
    contrast_factor = args.contrast
    max_dim = args.max_dim

    # Resolve target and source directories
    input_dir = os.path.dirname(os.path.abspath(inputfile))
    
    if args.dir:
        source_dir = os.path.abspath(args.dir)
    else:
        source_dir = os.path.join(input_dir, "source")
        
    # We place cache directory within source_dir
    cache_dir = os.path.join(source_dir, "cache")

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print("Making thumbnails...")
    # Find all source images (case-insensitive glob search)
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp')
    infiles = []
    for ext in valid_exts:
        infiles.extend(glob.glob(os.path.join(source_dir, ext)))
        infiles.extend(glob.glob(os.path.join(source_dir, ext.upper())))
    
    # Remove duplicates from glob
    infiles = list(set(infiles))

    for infile in infiles:
        try:
            filename = os.path.basename(infile)
            file_base, file_ext = os.path.splitext(filename)

            # Avoid making thumbnails of thumbnails
            if "_thumb" in file_base:
                continue

            thumb_path = os.path.join(cache_dir, f"{file_base}_thumb.jpg")
            if os.path.isfile(thumb_path):
                continue

            try:
                im = Image.open(infile)
            except Exception:
                continue

            # Standard EXIF orientation transpose
            im = ImageOps.exif_transpose(im)

            # Square crop from center
            w, h = im.size
            min_dim_sz = min(w, h)
            left = (w - min_dim_sz) // 2
            top = (h - min_dim_sz) // 2
            right = left + min_dim_sz
            bottom = top + min_dim_sz
            
            im = im.crop((left, top, right, bottom))
            im.thumbnail((thumb_size, thumb_size), LANCZOS)
            im.save(thumb_path, "JPEG")

        except Exception as err:
            print(f"Error preparing thumbnail for {infile}: {err}")

    # Collect cached thumbnails data
    thumbs_lib = {}
    thumbs_lib_avg = {}
    thumbs_lib_freq = {}
    
    cache_files = glob.glob(os.path.join(cache_dir, "*_thumb.jpg"))
    print(f"Collecting data on {len(cache_files)} thumbnails...")
    
    for infile in cache_files:
        filename = os.path.basename(infile)
        file_base, _ = os.path.splitext(filename)
        try:
            im = Image.open(infile)
            thumbs_lib[file_base] = im
            thumbs_lib_avg[file_base] = getAvgColor(im)
            thumbs_lib_freq[file_base] = getFrequent(im)
        except Exception as err:
            print(f"Error reading cached thumbnail {infile}: {err}")

    if not thumbs_lib:
        print("Error: No thumbnails found or created! Please ensure the source folder contains valid images.")
        return

    print("Resizing target...")
    try:
        target = Image.open(inputfile)
    except Exception as e:
        print(f"Error: Could not open target image {inputfile}: {e}")
        sys.exit(1)

    # Orientation fix for target
    target = ImageOps.exif_transpose(target)

    # Enhancing target image
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(target)
        target = enhancer.enhance(brightness_factor)
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(target)
        target = enhancer.enhance(contrast_factor)

    w, h = target.size
    scale_w = w / min(w, h)
    scale_h = h / min(w, h)
    
    x_tiles = int(scale_w * target_tiles)
    y_tiles = int(scale_h * target_tiles)
    
    target_thumb = target.copy()
    target_thumb.thumbnail((x_tiles, y_tiles), LANCZOS)
    
    # Calculate dimensions of the output image
    final_w = thumb_size * target_thumb.size[0]
    final_h = thumb_size * target_thumb.size[1]
    
    # Memory and resolution protection
    if final_w > max_dim or final_h > max_dim:
        orig_final_w, orig_final_h = final_w, final_h
        ratio = min(max_dim / final_w, max_dim / final_h)
        thumb_size = int(thumb_size * ratio)
        final_w = thumb_size * target_thumb.size[0]
        final_h = thumb_size * target_thumb.size[1]
        print(f"Warning: Dimensions would be {orig_final_w}x{orig_final_h} px. Scaled tile size down to {thumb_size}px (output: {final_w}x{final_h} px) to preserve memory.")

    last_choose = ""
    final = Image.new("RGB", (final_w, final_h), "black")
    print(f"Making collage ({target_thumb.size[0]} x {target_thumb.size[1]} tiles)...")

    # Match target pixels to best thumbnails
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
                continue

            choose = topN(bestPicks, N)
            k = random.randint(0, N - 1)

            attempts = 0
            while N > 1 and last_choose == choose[k][0] and attempts < 10:
                k = random.randint(0, N - 1)
                attempts += 1

            last_choose = choose[k][0]
            thumb_img = thumbs_lib[last_choose]
            
            if thumb_img.size != (thumb_size, thumb_size):
                thumb_img = thumb_img.resize((thumb_size, thumb_size), LANCZOS)

            final.paste(thumb_img, (i * thumb_size, j * thumb_size))

    # Blend original target image overlay for color reinforcement
    if blend_amt > 0:
        print(f"Blending with overlay (amount: {blend_amt})...")
        bigger = target.resize(final.size, LANCZOS)
        final = Image.blend(final, bigger, blend_amt)

    print(f"Saving collage to '{outputfile}'...")
    final.save(outputfile, "JPEG")
    print("Done!")

if __name__ == "__main__":
    main()