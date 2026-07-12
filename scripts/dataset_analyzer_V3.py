"""
===========================================================
AI Food Quality Platform
Dataset Analyzer v3.0

Author : Dia Maria

Features
--------
✔ Counts Fresh & Rotten images
✔ Calculates class distribution
✔ Detects image formats
✔ Calculates average image resolution
✔ Finds minimum & maximum resolution
✔ Detects corrupted images
✔ Detects invalid files
✔ Generates dataset statistics
✔ Saves report (Part 3)
===========================================================
"""

from pathlib import Path
from collections import Counter
from PIL import Image
from datetime import datetime


# ----------------------------------------------------------
# Supported Image Extensions
# ----------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ----------------------------------------------------------
# Get Dataset Categories
# ----------------------------------------------------------

def get_food_categories(dataset_path: Path):
    """
    Returns all food category folders.
    """

    return sorted(
        [
            folder
            for folder in dataset_path.iterdir()
            if folder.is_dir()
        ]
    )


# ----------------------------------------------------------
# Analyze Images in One Folder
# ----------------------------------------------------------

def analyze_folder(folder_path: Path):
    """
    Analyzes every image inside a folder.

    Returns

    image_count
    corrupted_images
    invalid_files
    format_counter
    resolutions
    """

    image_count = 0

    corrupted_images = 0

    invalid_files = 0

    format_counter = Counter()

    resolutions = []

    if not folder_path.exists():

        return (
            image_count,
            corrupted_images,
            invalid_files,
            format_counter,
            resolutions
        )

    for file in folder_path.iterdir():

        # Ignore directories

        if not file.is_file():
            continue

        extension = file.suffix.lower()

        if extension not in IMAGE_EXTENSIONS:

            invalid_files += 1
            continue

        image_count += 1

        format_counter[extension] += 1

        try:

            with Image.open(file) as img:

                width, height = img.size

                resolutions.append(
                    (
                        width,
                        height
                    )
                )

        except Exception:

            corrupted_images += 1

    return (

        image_count,

        corrupted_images,

        invalid_files,

        format_counter,

        resolutions
    )


# ----------------------------------------------------------
# Resolution Statistics
# ----------------------------------------------------------

def resolution_statistics(resolutions):
    """
    Returns

    Average Resolution

    Minimum Resolution

    Maximum Resolution
    """

    if len(resolutions) == 0:

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

    minimum = (

        min(widths),

        min(heights)

    )

    maximum = (

        max(widths),

        max(heights)

    )

    return (

        average,

        minimum,

        maximum

    )


# ----------------------------------------------------------
# Dataset Quality
# ----------------------------------------------------------

def dataset_status(fresh, rotten):

    if fresh == 0 or rotten == 0:

        return "Invalid", "Infinite"

    ratio = round(

        max(fresh, rotten) /
        min(fresh, rotten),

        2

    )

    if ratio <= 1.5:

        status = "Excellent"

    elif ratio <= 2:

        status = "Good"

    elif ratio <= 3:

        status = "Moderate"

    else:

        status = "Highly Imbalanced"

    return status, ratio


# ----------------------------------------------------------
# Analyze One Food Category
# ----------------------------------------------------------

def analyze_food(food_path: Path):
    """
    Performs complete analysis
    of one food category.
    """

    fresh_folder = food_path / "Fresh"

    rotten_folder = food_path / "Rotten"

    (
        fresh_count,
        fresh_corrupted,
        fresh_invalid,
        fresh_formats,
        fresh_resolutions
    ) = analyze_folder(fresh_folder)

    (
        rotten_count,
        rotten_corrupted,
        rotten_invalid,
        rotten_formats,
        rotten_resolutions
    ) = analyze_folder(rotten_folder)

    total_images = fresh_count + rotten_count

    if total_images == 0:

        fresh_percentage = 0

        rotten_percentage = 0

    else:

        fresh_percentage = round(

            fresh_count / total_images * 100,

            2

        )

        rotten_percentage = round(

            rotten_count / total_images * 100,

            2

        )

    all_formats = fresh_formats + rotten_formats

    all_resolutions = (

        fresh_resolutions +

        rotten_resolutions

    )

    average_resolution, minimum_resolution, maximum_resolution = (

        resolution_statistics(

            all_resolutions

        )

    )

    status, ratio = dataset_status(

        fresh_count,

        rotten_count

    )

    return {

        "food": food_path.name,

        "fresh": fresh_count,

        "rotten": rotten_count,

        "total": total_images,

        "fresh_percentage": fresh_percentage,

        "rotten_percentage": rotten_percentage,

        "status": status,

        "ratio": ratio,

        "formats": all_formats,

        "average_resolution": average_resolution,

        "minimum_resolution": minimum_resolution,

        "maximum_resolution": maximum_resolution,

        "corrupted":

            fresh_corrupted +

            rotten_corrupted,

        "invalid":

            fresh_invalid +

            rotten_invalid

    }

# ----------------------------------------------------------
# Build Report
# ----------------------------------------------------------

def generate_report(results):
    """
    Builds the complete dataset report as a string.
    """

    report = []

    report.append("\n")
    report.append("=" * 90)
    report.append("AI FOOD QUALITY PLATFORM")
    report.append("RGB DATASET ANALYSIS REPORT")
    report.append("=" * 90)
    report.append(
        f"Generated On : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )

    grand_fresh = 0
    grand_rotten = 0
    grand_total = 0

    total_corrupted = 0
    total_invalid = 0

    overall_formats = Counter()

    all_resolutions = []

    excellent = 0
    good = 0
    moderate = 0
    problematic = 0

    # -------------------------------
    # Individual Food Categories
    # -------------------------------

    for item in results:

        report.append("\n")
        report.append("-" * 90)

        report.append(f"Food Category : {item['food']}")

        report.append("-" * 90)

        report.append(f"Fresh Images      : {item['fresh']}")
        report.append(f"Rotten Images     : {item['rotten']}")
        report.append(f"Total Images      : {item['total']}")

        report.append(
            f"Fresh Percentage  : {item['fresh_percentage']:.2f}%"
        )

        report.append(
            f"Rotten Percentage : {item['rotten_percentage']:.2f}%"
        )

        report.append(
            f"Imbalance Ratio   : {item['ratio']}"
        )

        report.append(
            f"Dataset Status    : {item['status']}"
        )

        report.append("\nImage Formats")

        if len(item["formats"]) == 0:

            report.append("  None")

        else:

            for ext, count in sorted(item["formats"].items()):

                report.append(
                    f"  {ext.upper():<8} : {count}"
                )

        avg = item["average_resolution"]
        minimum = item["minimum_resolution"]
        maximum = item["maximum_resolution"]

        report.append(
            f"\nAverage Resolution : {avg[0]} x {avg[1]}"
        )

        report.append(
            f"Minimum Resolution : {minimum[0]} x {minimum[1]}"
        )

        report.append(
            f"Maximum Resolution : {maximum[0]} x {maximum[1]}"
        )

        report.append(
            f"Corrupted Images   : {item['corrupted']}"
        )

        report.append(
            f"Invalid Files      : {item['invalid']}"
        )

        # -------------------------

        grand_fresh += item["fresh"]
        grand_rotten += item["rotten"]
        grand_total += item["total"]

        total_corrupted += item["corrupted"]
        total_invalid += item["invalid"]

        overall_formats.update(item["formats"])

        if item["status"] == "Excellent":
            excellent += 1

        elif item["status"] == "Good":
            good += 1

        elif item["status"] == "Moderate":
            moderate += 1

        else:
            problematic += 1

        if item["average_resolution"] != (0, 0):

            all_resolutions.append(
                item["average_resolution"]
            )

    # ------------------------------------
    # Overall Resolution
    # ------------------------------------

    overall_average, overall_minimum, overall_maximum = (
        resolution_statistics(all_resolutions)
    )

    # ------------------------------------
    # Dataset Health Score
    # ------------------------------------

    health_score = 100

    health_score -= moderate * 5

    health_score -= problematic * 10

    health_score -= total_corrupted * 2

    health_score -= total_invalid

    if health_score < 0:
        health_score = 0

    # ------------------------------------
    # Overall Statistics
    # ------------------------------------

    report.append("\n")
    report.append("=" * 90)
    report.append("OVERALL DATASET STATISTICS")
    report.append("=" * 90)

    report.append(f"Food Categories     : {len(results)}")

    report.append(f"Fresh Images        : {grand_fresh}")

    report.append(f"Rotten Images       : {grand_rotten}")

    report.append(f"Total Images        : {grand_total}")

    if grand_total > 0:

        report.append(
            f"Fresh Percentage    : {(grand_fresh/grand_total)*100:.2f}%"
        )

        report.append(
            f"Rotten Percentage   : {(grand_rotten/grand_total)*100:.2f}%"
        )

    report.append("\nImage Formats")

    for ext, count in sorted(overall_formats.items()):

        report.append(
            f"  {ext.upper():<8} : {count}"
        )

    report.append(
        f"\nAverage Resolution : "
        f"{overall_average[0]} x {overall_average[1]}"
    )

    report.append(
        f"Minimum Resolution : "
        f"{overall_minimum[0]} x {overall_minimum[1]}"
    )

    report.append(
        f"Maximum Resolution : "
        f"{overall_maximum[0]} x {overall_maximum[1]}"
    )

    report.append(
        f"\nCorrupted Images : {total_corrupted}"
    )

    report.append(
        f"Invalid Files    : {total_invalid}"
    )

    report.append("\nDataset Quality")

    report.append(
        f"Excellent Categories : {excellent}"
    )

    report.append(
        f"Good Categories      : {good}"
    )

    report.append(
        f"Moderate Categories  : {moderate}"
    )

    report.append(
        f"Problematic          : {problematic}"
    )

    report.append(
        f"\nDataset Health Score : {health_score}/100"
    )

    report.append("=" * 90)

    return "\n".join(report)

# ----------------------------------------------------------
# Save Report
# ----------------------------------------------------------

def save_report(report_text):
    """
    Saves the report inside reports/dataset_report.txt
    """

    reports_folder = Path("reports")

    reports_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = reports_folder / "dataset_report.txt"

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report_text)

    return report_file


# ----------------------------------------------------------
# Print Report
# ----------------------------------------------------------

def print_report(report_text):
    """
    Prints the report to the console.
    """

    print(report_text)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    dataset_path = Path(
        "datasets/raw/original/fresh_vs_rotten_fruits/Fruit Freshness Dataset/Fruit Freshness Dataset"
    )

    if not dataset_path.exists():

        print("\nDataset folder not found.")
        print(dataset_path)

        return

    print("\nScanning dataset...")
    print("-" * 60)

    categories = get_food_categories(dataset_path)

    if len(categories) == 0:

        print("No food categories found.")

        return

    results = []

    for category in categories:

        print(f"Analyzing {category.name}...")

        results.append(
            analyze_food(category)
        )

    report = generate_report(results)

    print_report(report)

    report_path = save_report(report)

    print("\n" + "=" * 60)
    print("Dataset analysis completed successfully.")
    print(f"Report saved to : {report_path.resolve()}")
    print("=" * 60)


# ----------------------------------------------------------
# Run Program
# ----------------------------------------------------------

if __name__ == "__main__":

    main()