"""
=============================================================
AI Food Quality Platform
Dataset Analyzer V4 (Final)

Author : Dia Maria

Features
--------
✓ Class Distribution
✓ Resolution Analysis
✓ Image Format Analysis
✓ Corrupted Image Detection
✓ Duplicate Image Detection
✓ Blur Detection
✓ Brightness Analysis
✓ Tiny/Large Image Detection
✓ Dataset Health Score
✓ AI Recommendations
✓ Automatic Report Generation

=============================================================
"""

from pathlib import Path
from collections import Counter
from datetime import datetime

from PIL import Image
import imagehash
import cv2
import numpy as np



# =============================================================
# CONSTANTS
# =============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# Blur Threshold

BLUR_THRESHOLD = 100


# Brightness Thresholds

DARK_THRESHOLD = 50

BRIGHT_THRESHOLD = 205


# Image Size Thresholds

MIN_WIDTH = 100
MIN_HEIGHT = 100

MAX_WIDTH = 3000
MAX_HEIGHT = 3000


# Recommended Resize

RECOMMENDED_SIZE = (224, 224)



# =============================================================
# GET FOOD CATEGORIES
# =============================================================

def get_food_categories(dataset_path: Path):
    """
    Returns all food category folders.
    """

    return sorted(

        folder

        for folder in dataset_path.iterdir()

        if folder.is_dir()

    )



# =============================================================
# IMAGE HASH
# =============================================================

def calculate_hash(image_path):
    """
    Returns perceptual hash of image.
    """

    try:

        with Image.open(image_path) as img:

            return imagehash.phash(img)

    except Exception:

        return None



# =============================================================
# BLUR DETECTION
# =============================================================

def blur_score(image_path):
    """
    Returns variance of Laplacian.

    Lower value = Blurrier image.
    """

    image = cv2.imread(str(image_path))

    if image is None:

        return None

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    score = cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).var()

    return score



# =============================================================
# BRIGHTNESS
# =============================================================

def brightness(image_path):
    """
    Average brightness (0-255)
    """

    image = cv2.imread(str(image_path))

    if image is None:

        return None

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    return np.mean(gray)



# =============================================================
# IMAGE SIZE
# =============================================================

def image_size(image_path):
    """
    Returns

    width,

    height
    """

    try:

        with Image.open(image_path) as img:

            return img.size

    except Exception:

        return (0, 0)



# =============================================================
# IMAGE FORMAT
# =============================================================

def image_format(image_path):
    """
    Returns extension.
    """

    return image_path.suffix.lower()



# =============================================================
# IMAGE STATUS
# =============================================================

def classify_image(image_path):
    """
    Performs image quality analysis.

    Returns dictionary.
    """

    result = {

        "hash": None,

        "blur": None,

        "brightness": None,

        "width": 0,

        "height": 0,

        "tiny": False,

        "large": False,

        "dark": False,

        "bright": False,

        "blurry": False

    }

    # -------------------------
    # Hash
    # -------------------------

    result["hash"] = calculate_hash(image_path)

    # -------------------------
    # Blur
    # -------------------------

    blur = blur_score(image_path)

    result["blur"] = blur

    if blur is not None:

        if blur < BLUR_THRESHOLD:

            result["blurry"] = True

    # -------------------------
    # Brightness
    # -------------------------

    bright = brightness(image_path)

    result["brightness"] = bright

    if bright is not None:

        if bright < DARK_THRESHOLD:

            result["dark"] = True

        elif bright > BRIGHT_THRESHOLD:

            result["bright"] = True

    # -------------------------
    # Resolution
    # -------------------------

    width, height = image_size(image_path)

    result["width"] = width

    result["height"] = height

    if width < MIN_WIDTH or height < MIN_HEIGHT:

        result["tiny"] = True

    if width > MAX_WIDTH or height > MAX_HEIGHT:

        result["large"] = True

    return result



# =============================================================
# DUPLICATE DETECTOR
# =============================================================

def find_duplicates(image_hashes):
    """
    Finds duplicate images.

    Parameter

    image_hashes

    {

        hash : [

            file1,

            file2,

            ...

        ]

    }
    """

    duplicates = []

    for files in image_hashes.values():

        if len(files) > 1:

            duplicates.append(files)

    return duplicates

# =============================================================
# RESOLUTION STATISTICS
# =============================================================

def resolution_statistics(resolutions):
    """
    Returns

    Average Resolution

    Smallest Image

    Largest Image
    """

    if not resolutions:

        return (
            (0, 0),
            (0, 0),
            (0, 0)
        )

    widths = [w for w, h in resolutions]
    heights = [h for w, h in resolutions]

    average = (
        int(sum(widths) / len(widths)),
        int(sum(heights) / len(heights))
    )

    minimum = min(
        resolutions,
        key=lambda r: r[0] * r[1]
    )

    maximum = max(
        resolutions,
        key=lambda r: r[0] * r[1]
    )

    return average, minimum, maximum


# =============================================================
# ANALYZE FOLDER
# =============================================================

def analyze_folder(folder_path: Path):
    """
    Analyze one folder (Fresh or Rotten)
    """

    image_count = 0

    corrupted = []

    invalid_files = []

    blurry_images = []

    dark_images = []

    bright_images = []

    tiny_images = []

    large_images = []

    resolutions = []

    formats = Counter()

    image_hashes = {}

    if not folder_path.exists():

        return {

            "count": 0,

            "corrupted": [],

            "invalid": [],

            "blurry": [],

            "dark": [],

            "bright": [],

            "tiny": [],

            "large": [],

            "formats": Counter(),

            "hashes": {},

            "resolutions": []

        }

    # -----------------------------------------------------

    for file in sorted(folder_path.iterdir()):

        if not file.is_file():

            continue

        extension = file.suffix.lower()

        # Invalid file

        if extension not in IMAGE_EXTENSIONS:

            invalid_files.append(file.name)

            continue

        image_count += 1

        formats[extension] += 1

        # ---------------------------------------------

        try:

            analysis = classify_image(file)

        except Exception:

            corrupted.append(file.name)

            continue

        # ---------------------------------------------
        # Resolution
        # ---------------------------------------------

        resolutions.append(

            (

                analysis["width"],

                analysis["height"]

            )

        )

        # ---------------------------------------------
        # Duplicate
        # ---------------------------------------------

        image_hash = analysis["hash"]

        if image_hash is not None:

            image_hashes.setdefault(

                str(image_hash),

                []

            ).append(file.name)

        # ---------------------------------------------
        # Blur
        # ---------------------------------------------

        if analysis["blurry"]:

            blurry_images.append(

                {

                    "file": file.name,

                    "score": round(

                        analysis["blur"],

                        2

                    )

                }

            )

        # ---------------------------------------------
        # Brightness
        # ---------------------------------------------

        if analysis["dark"]:

            dark_images.append(

                {

                    "file": file.name,

                    "brightness": round(

                        analysis["brightness"],

                        2

                    )

                }

            )

        if analysis["bright"]:

            bright_images.append(

                {

                    "file": file.name,

                    "brightness": round(

                        analysis["brightness"],

                        2

                    )

                }

            )

        # ---------------------------------------------
        # Tiny / Large
        # ---------------------------------------------

        if analysis["tiny"]:

            tiny_images.append(

                {

                    "file": file.name,

                    "size":

                        f"{analysis['width']} x "

                        f"{analysis['height']}"

                }

            )

        if analysis["large"]:

            large_images.append(

                {

                    "file": file.name,

                    "size":

                        f"{analysis['width']} x "

                        f"{analysis['height']}"

                }

            )

    # -------------------------------------------------

    duplicates = find_duplicates(image_hashes)

    average, minimum, maximum = resolution_statistics(

        resolutions

    )

    return {

        "count": image_count,

        "corrupted": corrupted,

        "invalid": invalid_files,

        "blurry": blurry_images,

        "dark": dark_images,

        "bright": bright_images,

        "tiny": tiny_images,

        "large": large_images,

        "formats": formats,

        "duplicates": duplicates,

        "resolutions": resolutions,

        "average_resolution": average,

        "minimum_resolution": minimum,

        "maximum_resolution": maximum

    }


# =============================================================
# MERGE FORMAT COUNTERS
# =============================================================

def merge_formats(counter1, counter2):
    """
    Merge two format counters.
    """

    merged = Counter()

    merged.update(counter1)

    merged.update(counter2)

    return merged


# =============================================================
# MERGE LISTS
# =============================================================

def merge_lists(list1, list2):
    """
    Merge two lists.
    """

    return list1 + list2

