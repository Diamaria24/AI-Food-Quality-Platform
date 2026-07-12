"""
=============================================================
AI Food Quality Platform

Recommendation Engine

Author : Dia Maria
=============================================================
"""

from scripts.config.constants import (
    RECOMMENDED_RESIZE,
    MIN_IMAGES_PER_CLASS,
    STATUS_HIGHLY_IMBALANCED,
    STATUS_MODERATE,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW
)


# ============================================================
# CREATE RECOMMENDATION
# ============================================================

def create_recommendation(priority, title, message):
    """
    Creates a recommendation object.
    """

    return {
        "priority": priority,
        "title": title,
        "message": message
    }


# ============================================================
# CATEGORY RECOMMENDATIONS
# ============================================================

def generate_recommendations(category):
    """
    Generate recommendations for a single food category.
    """

    recommendations = []

    # -------------------------------------------------------
    # Dataset Balance
    # -------------------------------------------------------

    if category["status"] == STATUS_HIGHLY_IMBALANCED:

        if category["fresh"] < category["rotten"]:

            recommendations.append(
                create_recommendation(
                    PRIORITY_HIGH,
                    "Dataset Balance",
                    f"Collect more Fresh {category['food']} images."
                )
            )

        else:

            recommendations.append(
                create_recommendation(
                    PRIORITY_HIGH,
                    "Dataset Balance",
                    f"Collect more Rotten {category['food']} images."
                )
            )

    elif category["status"] == STATUS_MODERATE:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Dataset Balance",
                "Use data augmentation to improve class balance."
            )
        )

    # -------------------------------------------------------
    # Minimum Images
    # -------------------------------------------------------

    if category["fresh"] < MIN_IMAGES_PER_CLASS:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Fresh Dataset",
                "Fresh class has fewer images than recommended."
            )
        )

    if category["rotten"] < MIN_IMAGES_PER_CLASS:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Rotten Dataset",
                "Rotten class has fewer images than recommended."
            )
        )

    # -------------------------------------------------------
    # Duplicate Images
    # -------------------------------------------------------

    if category["duplicates"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_HIGH,
                "Duplicate Images",
                f"Remove {len(category['duplicates'])} duplicate image groups."
            )
        )

    # -------------------------------------------------------
    # Corrupted Images
    # -------------------------------------------------------

    if category["corrupted"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_HIGH,
                "Corrupted Images",
                f"Remove {len(category['corrupted'])} corrupted images."
            )
        )

    # -------------------------------------------------------
    # Invalid Files
    # -------------------------------------------------------

    if category["invalid"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_HIGH,
                "Invalid Files",
                f"Remove {len(category['invalid'])} invalid files."
            )
        )

    # -------------------------------------------------------
    # Blurry Images
    # -------------------------------------------------------

    if category["blurry"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Blurry Images",
                f"Review {len(category['blurry'])} blurry images."
            )
        )

    # -------------------------------------------------------
    # Tiny Images
    # -------------------------------------------------------

    if category["tiny"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Tiny Images",
                f"Replace {len(category['tiny'])} tiny images."
            )
        )

    # -------------------------------------------------------
    # Large Images
    # -------------------------------------------------------

    if category["large"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_LOW,
                "Large Images",
                "Resize large images before training."
            )
        )

    # -------------------------------------------------------
    # Dark Images
    # -------------------------------------------------------

    if category["dark"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_LOW,
                "Dark Images",
                f"Review {len(category['dark'])} dark images."
            )
        )

    # -------------------------------------------------------
    # Bright Images
    # -------------------------------------------------------

    if category["bright"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_LOW,
                "Bright Images",
                f"Review {len(category['bright'])} overexposed images."
            )
        )

    # -------------------------------------------------------
    # Resize Recommendation
    # -------------------------------------------------------

    recommendations.append(
        create_recommendation(
            PRIORITY_LOW,
            "Image Size",
            f"Resize images to {RECOMMENDED_RESIZE[0]}×{RECOMMENDED_RESIZE[1]}."
        )
    )

    return recommendations


# ============================================================
# OVERALL RECOMMENDATIONS
# ============================================================

def overall_recommendations(overall):
    """
    Generate recommendations for the complete dataset.
    """

    recommendations = []

    if overall["duplicates"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_HIGH,
                "Duplicate Images",
                f"Dataset contains {len(overall['duplicates'])} duplicate image groups."
            )
        )

    if overall["corrupted"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_HIGH,
                "Corrupted Images",
                f"Remove {len(overall['corrupted'])} corrupted images."
            )
        )

    if overall["invalid"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_HIGH,
                "Invalid Files",
                f"Remove {len(overall['invalid'])} invalid files."
            )
        )

    if overall["blurry"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Blurry Images",
                f"Review {len(overall['blurry'])} blurry images."
            )
        )

    if overall["tiny"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_MEDIUM,
                "Tiny Images",
                f"Replace {len(overall['tiny'])} tiny images."
            )
        )

    if overall["large"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_LOW,
                "Large Images",
                "Resize large images before preprocessing."
            )
        )

    if overall["dark"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_LOW,
                "Dark Images",
                "Improve illumination of dark images."
            )
        )

    if overall["bright"]:

        recommendations.append(
            create_recommendation(
                PRIORITY_LOW,
                "Bright Images",
                "Review overexposed images."
            )
        )

    recommendations.append(
        create_recommendation(
            PRIORITY_LOW,
            "Training",
            f"Recommended image size: {RECOMMENDED_RESIZE[0]}×{RECOMMENDED_RESIZE[1]}."
        )
    )

    recommendations.append(
        create_recommendation(
            PRIORITY_LOW,
            "Dataset",
            "Dataset is ready for preprocessing after the above issues are resolved."
        )
    )

    return recommendations


# ============================================================
# PRINT RECOMMENDATIONS
# ============================================================

def print_recommendations(recommendations):
    """
    Print recommendations grouped by priority.
    """

    if not recommendations:
        print("No recommendations.")
        return

    priorities = [PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]

    for priority in priorities:

        items = [
            r for r in recommendations
            if r["priority"] == priority
        ]

        if not items:
            continue

        print(f"\n{priority} PRIORITY")
        print("-" * 40)

        for item in items:
            print(f"• {item['message']}")