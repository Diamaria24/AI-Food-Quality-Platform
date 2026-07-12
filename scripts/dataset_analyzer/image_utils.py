"""
=============================================================
AI Food Quality Platform

Image Utility Functions

Author : Dia Maria
=============================================================
"""

from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from collections import Counter

from scripts.config.constants import (
    IMAGE_EXTENSIONS,
    BLUR_THRESHOLD,
    DARK_THRESHOLD,
    BRIGHT_THRESHOLD,
    MIN_WIDTH,
    MIN_HEIGHT,
    MAX_WIDTH,
    MAX_HEIGHT
)

# =============================================================
# READ IMAGE
# =============================================================

def read_image(image_path: Path):
    """
    Reads an image using OpenCV.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    numpy.ndarray | None
    """

    image = cv2.imread(str(image_path))

    return image

# =============================================================
# VALID IMAGE
# =============================================================

def is_valid_image(image_path: Path):
    """
    Checks whether the file extension
    is supported.
    """

    return image_path.suffix.lower() in IMAGE_EXTENSIONS

# =============================================================
# CORRUPTED IMAGE
# =============================================================

def is_corrupted(image_path: Path):
    """
    Returns True if image cannot be opened.
    """

    try:

        with Image.open(image_path) as img:

            img.verify()

        return False

    except Exception:

        return True
    
# =============================================================
# IMAGE SIZE
# =============================================================

def get_image_size(image_path: Path):
    """
    Returns image width and height.
    """

    try:

        with Image.open(image_path) as img:

            return img.size

    except Exception:

        return (0, 0)
    
# =============================================================
# IMAGE FORMAT
# =============================================================

def get_image_format(image_path: Path):
    """
    Returns file extension.
    """

    return image_path.suffix.lower()

# =============================================================
# IMAGE HASH
# =============================================================

def calculate_hash(image_path: Path):
    """
    Calculates perceptual hash.
    """

    try:

        with Image.open(image_path) as img:

            return imagehash.phash(img)

    except Exception:

        return None
    
# =============================================================
# BLUR SCORE
# =============================================================

def get_blur_score(image):
    """
    Variance of Laplacian.

    Higher score = Sharper image.
    """

    if image is None:

        return 0

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    score = cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).var()

    return float(score)

# =============================================================
# BLUR DETECTION
# =============================================================

def is_blurry(image):
    """
    Returns True if image is blurry.
    """

    return get_blur_score(image) < BLUR_THRESHOLD

# =============================================================
# BRIGHTNESS
# =============================================================

def get_brightness(image):
    """
    Average brightness (0-255)
    """

    if image is None:

        return 0

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )

    return float(np.mean(gray))

# =============================================================
# DARK IMAGE
# =============================================================

def is_dark(image):

    return get_brightness(image) < DARK_THRESHOLD

# =============================================================
# BRIGHT IMAGE
# =============================================================

def is_bright(image):

    return get_brightness(image) > BRIGHT_THRESHOLD

# =============================================================
# TINY IMAGE
# =============================================================

def is_tiny(width, height):

    return (

        width < MIN_WIDTH

        or

        height < MIN_HEIGHT

    )

# =============================================================
# LARGE IMAGE
# =============================================================

def is_large(width, height):

    return (

        width > MAX_WIDTH

        or

        height > MAX_HEIGHT

    )

# =============================================================
# ANALYZE SINGLE IMAGE
# =============================================================

def analyze_image(image_path: Path):
    """
    Performs complete analysis of one image.

    Returns
    -------
    dict
    """

    result = {

        "file": image_path.name,

        "path": image_path,

        "format": get_image_format(image_path),

        "width": 0,

        "height": 0,

        "hash": None,

        "blur_score": 0,

        "brightness": 0,

        "corrupted": False,

        "blurry": False,

        "dark": False,

        "bright": False,

        "tiny": False,

        "large": False

    }

    # -----------------------------------------
    # Corrupted Image
    # -----------------------------------------

    if is_corrupted(image_path):

        result["corrupted"] = True

        return result

    # -----------------------------------------
    # Read Image
    # -----------------------------------------

    image = read_image(image_path)

    if image is None:

        result["corrupted"] = True

        return result

    # -----------------------------------------
    # Resolution
    # -----------------------------------------

    width, height = get_image_size(image_path)

    result["width"] = width

    result["height"] = height

    # -----------------------------------------
    # Hash
    # -----------------------------------------

    result["hash"] = calculate_hash(image_path)

    # -----------------------------------------
    # Blur
    # -----------------------------------------

    blur = get_blur_score(image)

    result["blur_score"] = round(

        blur,

        2

    )

    result["blurry"] = is_blurry(image)

    # -----------------------------------------
    # Brightness
    # -----------------------------------------

    brightness = get_brightness(image)

    result["brightness"] = round(

        brightness,

        2

    )

    result["dark"] = is_dark(image)

    result["bright"] = is_bright(image)

    # -----------------------------------------
    # Resolution Checks
    # -----------------------------------------

    result["tiny"] = is_tiny(

        width,

        height

    )

    result["large"] = is_large(

        width,

        height

    )

    return result


# =============================================================
# FIND DUPLICATE IMAGES
# =============================================================

def find_duplicates(image_results):
    """
    Finds duplicate images using perceptual hash.

    Parameters
    ----------
    image_results : list

    Returns
    -------
    list
    """

    hash_map = {}

    for image in image_results:

        if image["corrupted"]:

            continue

        image_hash = image["hash"]

        if image_hash is None:

            continue

        hash_map.setdefault(

            str(image_hash),

            []

        ).append(

            image["path"]

        )

    duplicates = []

    for files in hash_map.values():

        if len(files) > 1:

            duplicates.append(files)

    return duplicates


# =============================================================
# SCAN FOLDER
# =============================================================

def scan_folder(folder_path: Path):
    """
    Scans every image inside a folder.

    Returns
    -------
    list
    """

    results = []

    if not folder_path.exists():

        return results

    for file in sorted(folder_path.iterdir()):

        if not file.is_file():

            continue

        if not is_valid_image(file):

            continue

        results.append(

            analyze_image(file)

        )

    return results


# =============================================================
# GET INVALID FILES
# =============================================================

def get_invalid_files(folder_path: Path):
    """
    Returns all unsupported files.
    """

    invalid = []

    if not folder_path.exists():

        return invalid

    for file in sorted(folder_path.iterdir()):

        if not file.is_file():

            continue

        if not is_valid_image(file):

            invalid.append(file.name)

    return invalid


# =============================================================
# FILTER FUNCTIONS
# =============================================================

def get_corrupted_images(images):

    return [

        image

        for image in images

        if image["corrupted"]

    ]


def get_blurry_images(images):

    return [

        image

        for image in images

        if image["blurry"]

    ]


def get_dark_images(images):

    return [

        image

        for image in images

        if image["dark"]

    ]


def get_bright_images(images):

    return [

        image

        for image in images

        if image["bright"]

    ]


def get_tiny_images(images):

    return [

        image

        for image in images

        if image["tiny"]

    ]


def get_large_images(images):

    return [

        image

        for image in images

        if image["large"]

    ]

# =============================================================
# IMAGE FORMAT STATISTICS
# =============================================================

def format_statistics(images):
    """
    Counts image formats.

    Returns
    -------
    Counter
    """

    formats = Counter()

    for image in images:

        if image["corrupted"]:

            continue

        formats[image["format"]] += 1

    return formats


# =============================================================
# RESOLUTION LIST
# =============================================================

def resolution_list(images):
    """
    Returns list of (width,height)
    """

    resolutions = []

    for image in images:

        if image["corrupted"]:

            continue

        resolutions.append(

            (

                image["width"],

                image["height"]

            )

        )

    return resolutions


# =============================================================
# IMAGE COUNT
# =============================================================

def image_count(images):

    return len(images)


# =============================================================
# VALID IMAGE COUNT
# =============================================================

def valid_image_count(images):

    return sum(

        not image["corrupted"]

        for image in images

    )


# =============================================================
# PRINT IMAGE SUMMARY
# =============================================================

def print_image_summary(images):
    """
    Debug helper.
    """

    print("\nImage Summary")
    print("-" * 40)

    print(f"Images      : {image_count(images)}")
    print(f"Valid       : {valid_image_count(images)}")
    print(f"Corrupted   : {len(get_corrupted_images(images))}")
    print(f"Blurry      : {len(get_blurry_images(images))}")
    print(f"Dark        : {len(get_dark_images(images))}")
    print(f"Bright      : {len(get_bright_images(images))}")
    print(f"Tiny        : {len(get_tiny_images(images))}")
    print(f"Large       : {len(get_large_images(images))}")
    print(f"Duplicates  : {len(find_duplicates(images))}")

