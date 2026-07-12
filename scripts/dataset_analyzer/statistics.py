"""
=============================================================
AI Food Quality Platform

Dataset Statistics Module

Author : Dia Maria
=============================================================
"""

from collections import Counter

from scripts.config.constants import (
    INITIAL_HEALTH_SCORE,
    HIGHLY_IMBALANCED_PENALTY,
    MODERATE_IMBALANCE_PENALTY,
    DUPLICATE_IMAGE_PENALTY,
    CORRUPTED_IMAGE_PENALTY,
    BLURRY_IMAGE_PENALTY,
    INVALID_FILE_PENALTY,
    TINY_IMAGE_PENALTY,
    MIN_IMAGES_PER_CLASS,
    STATUS_EXCELLENT,
    STATUS_GOOD,
    STATUS_MODERATE,
    STATUS_HIGHLY_IMBALANCED,
    STATUS_INVALID
)

# ============================================================
# PERCENTAGE
# ============================================================

def percentage(part, total):
    """
    Returns percentage.
    """

    if total == 0:
        return 0.0

    return round((part / total) * 100, 2)

# ============================================================
# RESOLUTION STATISTICS
# ============================================================

def resolution_statistics(resolutions):
    """
    Returns

    Average

    Minimum

    Maximum Resolution
    """

    if not resolutions:

        return (

            (0, 0),

            (0, 0),

            (0, 0)

        )

    widths = [

        width

        for width, height in resolutions

    ]

    heights = [

        height

        for width, height in resolutions

    ]

    average = (

        int(sum(widths) / len(widths)),

        int(sum(heights) / len(heights))

    )

    minimum = min(

        resolutions,

        key=lambda size: size[0] * size[1]

    )

    maximum = max(

        resolutions,

        key=lambda size: size[0] * size[1]

    )

    return (

        average,

        minimum,

        maximum

    )

# ============================================================
# DATASET BALANCE
# ============================================================

def dataset_status(fresh_count, rotten_count):
    """
    Returns dataset quality and imbalance ratio.
    """

    if fresh_count == 0 or rotten_count == 0:

        return (

            STATUS_INVALID,

            float("inf")

        )

    ratio = round(

        max(fresh_count, rotten_count)

        /

        min(fresh_count, rotten_count),

        2

    )

    if ratio <= 1.5:

        status = STATUS_EXCELLENT

    elif ratio <= 2:

        status = STATUS_GOOD

    elif ratio <= 3:

        status = STATUS_MODERATE

    else:

        status = STATUS_HIGHLY_IMBALANCED

    return status, ratio

# ============================================================
# IMAGE FORMAT STATISTICS
# ============================================================

def format_statistics(images):
    """
    Counts image formats.
    """

    formats = Counter()

    for image in images:

        if image["corrupted"]:

            continue

        formats[image["format"]] += 1

    return formats

# ============================================================
# DATASET HEALTH SCORE
# ============================================================

def dataset_health_score(stats):
    """
    Calculates dataset quality score (0-100).
    Uses percentages instead of absolute counts.
    """

    score = INITIAL_HEALTH_SCORE

    total = stats["total"]

    if total == 0:
        return 0

    # -------------------------------------------------
    # Dataset imbalance
    # -------------------------------------------------

    if stats["status"] == STATUS_HIGHLY_IMBALANCED:

        score -= 15

    elif stats["status"] == STATUS_MODERATE:

        score -= 8

    # -------------------------------------------------
    # Percentages
    # -------------------------------------------------

    duplicate_percent = (
        len(stats["duplicates"]) / total
    ) * 100

    corrupted_percent = (
        len(stats["corrupted"]) / total
    ) * 100

    blurry_percent = (
        len(stats["blurry"]) / total
    ) * 100

    invalid_percent = (
        len(stats["invalid"]) / total
    ) * 100

    bright_percent = (
        len(stats["bright"]) / total
    ) * 100

    dark_percent = (
        len(stats["dark"]) / total
    ) * 100

    # -------------------------------------------------
    # Duplicate images
    # -------------------------------------------------

    if duplicate_percent > 20:

        score -= 15

    elif duplicate_percent > 10:

        score -= 8

    elif duplicate_percent > 0:

        score -= 3

    # -------------------------------------------------
    # Corrupted images
    # -------------------------------------------------

    if corrupted_percent > 5:

        score -= 20

    elif corrupted_percent > 0:

        score -= 10

    # -------------------------------------------------
    # Invalid files
    # -------------------------------------------------

    if invalid_percent > 5:

        score -= 10

    elif invalid_percent > 0:

        score -= 5

    # -------------------------------------------------
    # Blurry images
    # -------------------------------------------------

    if blurry_percent > 20:

        score -= 15

    elif blurry_percent > 10:

        score -= 8

    elif blurry_percent > 0:

        score -= 3

    # -------------------------------------------------
    # Bright / Dark images
    # -------------------------------------------------

    if bright_percent > 30:

        score -= 8

    elif bright_percent > 10:

        score -= 4

    if dark_percent > 30:

        score -= 8

    elif dark_percent > 10:

        score -= 4

    # -------------------------------------------------
    # Minimum images
    # -------------------------------------------------

    if stats["fresh"] < MIN_IMAGES_PER_CLASS:

        score -= 5

    if stats["rotten"] < MIN_IMAGES_PER_CLASS:

        score -= 5

    # -------------------------------------------------

    score = max(score, 0)
    score = min(score, 100)

    return score

def dataset_grade(score):
    """
    Converts health score to grade.
    """

    if score >= 85:
        return "A"

    elif score >= 70:
        return "B"

    elif score >= 55:
        return "C"

    elif score >= 40:
        return "D"

    return "F"

# ============================================================
# OVERALL DATASET STATISTICS
# ============================================================

def overall_statistics(results):
    """
    Combines statistics from every food category.
    """

    overall = {

        "fresh": 0,

        "rotten": 0,

        "total": 0,

        "duplicates": [],

        "corrupted": [],

        "blurry": [],

        "dark": [],

        "bright": [],

        "tiny": [],

        "large": [],

        "invalid": [],

        "formats": Counter(),

        "resolutions": []

    }

    for category in results:

        overall["fresh"] += category["fresh"]

        overall["rotten"] += category["rotten"]

        overall["total"] += category["total"]

        overall["duplicates"].extend(

            category["duplicates"]

        )

        overall["corrupted"].extend(

            category["corrupted"]

        )

        overall["blurry"].extend(

            category["blurry"]

        )

        overall["dark"].extend(

            category["dark"]

        )

        overall["bright"].extend(

            category["bright"]

        )

        overall["tiny"].extend(

            category["tiny"]

        )

        overall["large"].extend(

            category["large"]

        )

        overall["invalid"].extend(

            category["invalid"]

        )

        overall["formats"].update(

            category["formats"]

        )

        overall["resolutions"].extend(

            category["resolutions"]

        )

    overall["fresh_percentage"] = percentage(

        overall["fresh"],

        overall["total"]

    )

    overall["rotten_percentage"] = percentage(

        overall["rotten"],

        overall["total"]

    )

    (
        overall["average_resolution"],

        overall["minimum_resolution"],

        overall["maximum_resolution"]

    ) = resolution_statistics(

        overall["resolutions"]

    )

    return overall

# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(results):
    """
    Returns summary of category quality.
    """

    summary = {

        "excellent": 0,

        "good": 0,

        "moderate": 0,

        "problematic": 0

    }

    for category in results:

        status = category["status"]

        if status == STATUS_EXCELLENT:

            summary["excellent"] += 1

        elif status == STATUS_GOOD:

            summary["good"] += 1

        elif status == STATUS_MODERATE:

            summary["moderate"] += 1

        else:

            summary["problematic"] += 1

    return summary

