"""
=============================================================
AI Food Quality Platform

Dataset Analyzer

Author : Dia Maria
=============================================================
"""

from pathlib import Path

from .image_utils import (
    scan_folder,
    get_invalid_files,
    find_duplicates,
    format_statistics,
    resolution_list,
)

from .statistics import (
    percentage,
    dataset_status,
    dataset_health_score,
    dataset_grade,
    resolution_statistics,
    overall_statistics,
    dataset_summary,
)

from .recommendations import (
    generate_recommendations,
    overall_recommendations,
)

from .report import (
    create_report,
)


# ============================================================
# ANALYZE CLASS
# ============================================================

def analyze_class(folder: Path):
    """
    Analyze a single class folder
    (Fresh or Rotten).

    Parameters
    ----------
    folder : Path

    Returns
    -------
    dict
    """

    # ---------------------------------------------
    # Scan all images
    # ---------------------------------------------

    images = scan_folder(folder)

    # ---------------------------------------------
    # Invalid files
    # ---------------------------------------------

    invalid = get_invalid_files(folder)

    # ---------------------------------------------
    # Duplicate images
    # ---------------------------------------------

    duplicates = find_duplicates(images)

    # ---------------------------------------------
    # Categorize images
    # ---------------------------------------------

    corrupted = [
        image
        for image in images
        if image["corrupted"]
    ]

    blurry = [
        image
        for image in images
        if image["blurry"]
    ]

    dark = [
        image
        for image in images
        if image["dark"]
    ]

    bright = [
        image
        for image in images
        if image["bright"]
    ]

    tiny = [
        image
        for image in images
        if image["tiny"]
    ]

    large = [
        image
        for image in images
        if image["large"]
    ]

    # ---------------------------------------------
    # Return analysis
    # ---------------------------------------------

    return {

        "images": images,

        "count": len(images),

        "duplicates": duplicates,

        "invalid": invalid,

        "corrupted": corrupted,

        "blurry": blurry,

        "dark": dark,

        "bright": bright,

        "tiny": tiny,

        "large": large,

        "formats": format_statistics(images),

        "resolutions": resolution_list(images)

    }

# ============================================================
# ANALYZE FOOD CATEGORY
# ============================================================

def analyze_category(category_path: Path):
    """
    Analyze one food category
    (Apple, Banana, Tomato, etc.)
    """

    # -------------------------------------------------------
    # Analyze Fresh and Rotten folders
    # -------------------------------------------------------

    fresh = analyze_class(
        category_path / "Fresh"
    )

    rotten = analyze_class(
        category_path / "Rotten"
    )

    # -------------------------------------------------------
    # Basic Counts
    # -------------------------------------------------------

    fresh_count = fresh["count"]

    rotten_count = rotten["count"]

    total = fresh_count + rotten_count

    # -------------------------------------------------------
    # Dataset Status
    # -------------------------------------------------------

    status, ratio = dataset_status(
        fresh_count,
        rotten_count
    )

    # -------------------------------------------------------
    # Merge Image Formats
    # -------------------------------------------------------

    formats = fresh["formats"].copy()

    formats.update(
        rotten["formats"]
    )

    # -------------------------------------------------------
    # Merge Resolution Lists
    # -------------------------------------------------------

    resolutions = (
        fresh["resolutions"] +
        rotten["resolutions"]
    )

    (
        average_resolution,
        minimum_resolution,
        maximum_resolution
    ) = resolution_statistics(
        resolutions
    )

    # -------------------------------------------------------
    # Build Category Dictionary
    # -------------------------------------------------------

    category = {

        "food": category_path.name,

        "fresh": fresh_count,

        "rotten": rotten_count,

        "total": total,

        "fresh_percentage": percentage(
            fresh_count,
            total
        ),

        "rotten_percentage": percentage(
            rotten_count,
            total
        ),

        "status": status,

        "imbalance_ratio": ratio,

        "duplicates": (
            fresh["duplicates"] +
            rotten["duplicates"]
        ),

        "invalid": (
            fresh["invalid"] +
            rotten["invalid"]
        ),

        "corrupted": (
            fresh["corrupted"] +
            rotten["corrupted"]
        ),

        "blurry": (
            fresh["blurry"] +
            rotten["blurry"]
        ),

        "dark": (
            fresh["dark"] +
            rotten["dark"]
        ),

        "bright": (
            fresh["bright"] +
            rotten["bright"]
        ),

        "tiny": (
            fresh["tiny"] +
            rotten["tiny"]
        ),

        "large": (
            fresh["large"] +
            rotten["large"]
        ),

        "formats": formats,

        "resolutions": resolutions,

        "average_resolution": average_resolution,

        "minimum_resolution": minimum_resolution,

        "maximum_resolution": maximum_resolution

    }

    # -------------------------------------------------------
    # Dataset Health Score
    # -------------------------------------------------------

    category["health_score"] = dataset_health_score(
        category
    )

    category["grade"] = dataset_grade(
    category["health_score"]
    )

    # -------------------------------------------------------
    # AI Recommendations
    # -------------------------------------------------------

    category["recommendations"] = generate_recommendations(
        category
    )

    return category

# ============================================================
# ANALYZE COMPLETE DATASET
# ============================================================

def analyze_dataset(dataset_path: Path):
    """
    Analyze the complete RGB dataset.

    Parameters
    ----------
    dataset_path : Path

    Returns
    -------
    dict
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():

        raise FileNotFoundError(

            f"Dataset folder not found:\n{dataset_path}"

        )

    print()

    print("=" * 70)

    print("Scanning dataset...")

    print("=" * 70)

    categories = []

    # -------------------------------------------------------
    # Analyze each food category
    # -------------------------------------------------------

    for folder in sorted(dataset_path.iterdir()):

        if not folder.is_dir():

            continue

        print(f"[INFO] Analyzing {folder.name.title()}...")

        category = analyze_category(folder)

        categories.append(category)

    print()

    print("=" * 70)

    print("Dataset analysis completed.")

    print("=" * 70)

    # -------------------------------------------------------
    # Overall Statistics
    # -------------------------------------------------------

    overall = overall_statistics(
        categories
    )

    # -------------------------------------------------------
    # Dataset Summary
    # -------------------------------------------------------

    summary = dataset_summary(
        categories
    )

    # -------------------------------------------------------
    # Overall AI Recommendations
    # -------------------------------------------------------

    recommendations = overall_recommendations(
        overall
    )

    # -------------------------------------------------------
    # Generate Report
    # -------------------------------------------------------

    report = create_report(

        categories,

        overall,

        recommendations

    )

    # -------------------------------------------------------
    # Return Results
    # -------------------------------------------------------

    return {

        "categories": categories,

        "overall": overall,

        "summary": summary,

        "recommendations": recommendations,

        "report": report

    }

