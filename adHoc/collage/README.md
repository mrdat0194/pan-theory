# Photo Collage Generator (`adHoc/collage`)

A high-performance Python script to generate beautiful photo collages from a directory of source images, matching the colors and structure of a target image.

Refactored for cross-platform stability (Windows/Linux/macOS), modern CLI parameter parsing, vectorized image calculations, and memory protection safeguards.

---

## 🚀 How to Use

### Prerequisites
Install the required image processing library:
```bash
pip install Pillow
```

### Basic Command Line Usage
```bash
python adHoc/collage/collage.py -i <target_image> -d <source_images_dir> -o <output_collage_path>
```

---

## 🛠️ CLI Options

| Flag | Long Option | Default | Description |
| :--- | :--- | :--- | :--- |
| **`-i`** | `--ifile` | *Required* | Path to the target image that the collage will represent. |
| **`-d`** | `--dir` | `source/` next to `-i` | Directory containing the source images to use as tiles. |
| **`-o`** | `--ofile` | `outfile.jpeg` | Output collage filename. |
| **`-t`** | `--tilesize` | `200` | Target size of each square thumbnail tile (in pixels). |
| **`-n`** | `--numtiles` | `200` | Number of tiles along the minimum dimension (defines density/resolution). |
| **`-b`** | `--blend-amt` | `0.2` | Overlay blending amount of target image [`0.0` to `1.0`]. Lower values yield more vibrant tiles; higher values overlay original shapes strongly. |
| **`-B`** | `--brightness`| `1.1` | Brightness enhancement factor applied to the target image. |
| **`-C`** | `--contrast` | `1.1` | Contrast enhancement factor applied to the target image. |
| **`-m`** | `--max-dim` | `8000` | Maximum final dimension in pixels. Protects system memory from thrashing on large collages. |

---

## 💡 Examples

### 1. Simple Collage
Use all default configurations (200x200 tiles, blend overlay 0.2):
```bash
python adHoc/collage/collage.py -i adHoc/collage/dat.jpg -d adHoc/collage/img -o adHoc/collage/collage_output.jpg
```

### 2. High Density and Vibrant Tiles (No Blending)
Create a high-density collage with 300 tiles along the minimum dimension, 100px tile sizes, and zero blending to keep the source tile details completely raw and un-washed:
```bash
python adHoc/collage/collage.py -i adHoc/collage/dat.jpg -d adHoc/collage/img -o adHoc/collage/raw_tiles.jpg -n 300 -t 100 -b 0.0
```

```bash
python adHoc/collage/collage.py -i adHoc/collage/dat.jpg -d adHoc/collage/img -o adHoc/collage/collage_test_high.jpg -n 150 -t 50 -b 0.45
```
---

## 📈 Troubleshooting & Performance Tips

*   **Why does my output look dull or washed out?**
    If the blending amount (`-b` / `--blend-amt`) is set high (e.g. `0.7`), the original image is strongly blended over the collage, reducing contrast and detail. Set `-b 0.1` or `-b 0.0` for highly defined, vibrant individual tiles.
*   **Thumbnail Caching:**
    Thumbnails are automatically cropped to squares, resized, and cached inside a `cache/` directory inside your source directory (e.g. `img/cache/`). Subsequent runs using the same source directory will instantly load cached thumbnails without reprocessing.
*   **Memory Safeguard (`--max-dim`):**
    If you ask for a high tile density and high tile size, the script will output a warning and dynamically scale the tile size down to prevent huge memory spikes (e.g. keeping width/height within `8000` pixels).
