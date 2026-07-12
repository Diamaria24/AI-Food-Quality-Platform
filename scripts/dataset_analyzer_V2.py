"""
AI Food Quality Platform
Dataset Analyzer v2.0

Author: Dia Maria
"""

from pathlib import Path


# --------------------------------------------------
# Supported Image Extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# --------------------------------------------------
# Get Food Categories
# --------------------------------------------------

def get_food_categories(dataset_path: Path):
    """
    Returns all food category folders.
    """

    return sorted(
        [item for item in dataset_path.iterdir() if item.is_dir()]
    )


# --------------------------------------------------
# Count Images
# --------------------------------------------------

def count_images(folder: Path):

    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------
# Analyze One Food Category
# --------------------------------------------------

def analyze_food(food_path: Path):

    fresh_folder = food_path / "Fresh"
    rotten_folder = food_path / "Rotten"

    fresh_exists = fresh_folder.exists()
    rotten_exists = rotten_folder.exists()

    fresh_count = count_images(fresh_folder)
    rotten_count = count_images(rotten_folder)

    total = fresh_count + rotten_count

    if total > 0:

        fresh_percent = (fresh_count / total) * 100
        rotten_percent = (rotten_count / total) * 100

    else:

        fresh_percent = 0
        rotten_percent = 0

    if min(fresh_count, rotten_count) == 0:

        imbalance_ratio = "Infinite"

    else:

        imbalance_ratio = round(
            max(fresh_count, rotten_count) /
            min(fresh_count, rotten_count),
            2
        )

    # Dataset Quality

    if not fresh_exists or not rotten_exists:

        status = "Missing Folder"

    elif total == 0:

        status = "Empty Dataset"

    elif imbalance_ratio == "Infinite":

        status = "Invalid"

    elif imbalance_ratio <= 1.5:

        status = "Excellent"

    elif imbalance_ratio <= 2:

        status = "Good"

    elif imbalance_ratio <= 3:

        status = "Moderate"

    else:

        status = "Highly Imbalanced"

    return {

        "food": food_path.name,

        "fresh": fresh_count,
        "rotten": rotten_count,
        "total": total,

        "fresh_percent": fresh_percent,
        "rotten_percent": rotten_percent,

        "imbalance_ratio": imbalance_ratio,

        "status": status

    }


# --------------------------------------------------
# Print Report
# --------------------------------------------------

def print_summary(results):

    print("\n")
    print("=" * 90)
    print("RGB DATASET ANALYSIS REPORT")
    print("=" * 90)

    grand_fresh = 0
    grand_rotten = 0
    grand_total = 0

    excellent = 0
    good = 0
    moderate = 0
    bad = 0

    for item in results:

        print(f"\nFood Category : {item['food']}")
        print("-" * 40)

        print(f"Fresh Images      : {item['fresh']}")
        print(f"Rotten Images     : {item['rotten']}")
        print(f"Total Images      : {item['total']}")

        print(f"Fresh Percentage  : {item['fresh_percent']:.2f}%")
        print(f"Rotten Percentage : {item['rotten_percent']:.2f}%")

        print(f"Imbalance Ratio   : {item['imbalance_ratio']}")
        print(f"Dataset Status    : {item['status']}")

        grand_fresh += item["fresh"]
        grand_rotten += item["rotten"]
        grand_total += item["total"]

        if item["status"] == "Excellent":
            excellent += 1

        elif item["status"] == "Good":
            good += 1

        elif item["status"] == "Moderate":
            moderate += 1

        else:
            bad += 1

    print("\n")
    print("=" * 90)
    print("OVERALL DATASET STATISTICS")
    print("=" * 90)

    print(f"Total Fresh Images  : {grand_fresh}")
    print(f"Total Rotten Images : {grand_rotten}")
    print(f"Grand Total Images  : {grand_total}")

    if grand_total > 0:

        print(f"Fresh Percentage    : {(grand_fresh/grand_total)*100:.2f}%")
        print(f"Rotten Percentage   : {(grand_rotten/grand_total)*100:.2f}%")

    print("\nDataset Quality Summary")
    print("-----------------------")

    print(f"Excellent Categories      : {excellent}")
    print(f"Good Categories           : {good}")
    print(f"Moderate Categories       : {moderate}")
    print(f"Problematic Categories    : {bad}")

    print("=" * 90)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    dataset_path = Path(
        "datasets/raw/original/fresh_vs_rotten_fruits/Fruit Freshness Dataset/Fruit Freshness Dataset"
    )

    if not dataset_path.exists():

        print("Dataset folder not found.")
        return

    categories = get_food_categories(dataset_path)

    results = []

    for category in categories:

        results.append(
            analyze_food(category)
        )

    print_summary(results)


if __name__ == "__main__":

    main()