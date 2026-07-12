"""
=============================================================
AI Food Quality Platform

Report Generator

Author : Dia Maria
=============================================================
"""

from datetime import datetime

from scripts.config.constants import (
    LINE_LENGTH,
    DATE_FORMAT,
    DATASET_REPORT,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW
)


# ============================================================
# SEPARATOR
# ============================================================

def separator(character="="):
    """
    Returns a separator line.
    """

    return character * LINE_LENGTH


# ============================================================
# REPORT HEADER
# ============================================================

def report_header():
    """
    Creates the report header.
    """

    return [

        separator(),

        "AI FOOD QUALITY PLATFORM",

        "RGB DATASET ANALYSIS REPORT",

        separator(),

        f"Generated On : {datetime.now().strftime(DATE_FORMAT)}",

        ""

    ]


# ============================================================
# CATEGORY REPORT
# ============================================================

def category_report(category):
    """
    Generates summary for one food category.
    """

    return [

        separator("-"),

        f"Food Category : {category['food'].title()}",

        separator("-"),

        f"Fresh Images      : {category['fresh']}",

        f"Rotten Images     : {category['rotten']}",

        f"Total Images      : {category['total']}",

        f"Fresh Percentage  : {category['fresh_percentage']}%",

        f"Rotten Percentage : {category['rotten_percentage']}%",

        f"Imbalance Ratio   : {category['imbalance_ratio']}",

        f"Dataset Status    : {category['status']}",

        f"Health Score      : {category['health_score']}/100",

        f"Dataset Grade    : {category['grade']}",

        ""

    ]


# ============================================================
# RESOLUTION SECTION
# ============================================================

def resolution_section(category):
    """
    Prints resolution statistics.
    """

    average = category["average_resolution"]

    minimum = category["minimum_resolution"]

    maximum = category["maximum_resolution"]

    return [

        "Resolution Statistics",

        f"Average Resolution : {average[0]} x {average[1]}",

        f"Minimum Resolution : {minimum[0]} x {minimum[1]}",

        f"Maximum Resolution : {maximum[0]} x {maximum[1]}",

        ""

    ]


# ============================================================
# IMAGE FORMAT SECTION
# ============================================================

def format_section(category):
    """
    Displays image format statistics.
    """

    lines = [

        "Image Formats"

    ]

    for extension, count in sorted(

        category["formats"].items()

    ):

        lines.append(

            f"  {extension.upper():<8}: {count}"

        )

    lines.append("")

    return lines


# ============================================================
# ISSUE SECTION
# ============================================================

def issue_section(category):
    """
    Displays detected dataset issues.
    """

    return [

        "Detected Issues",

        f"Duplicate Groups : {len(category['duplicates'])}",

        f"Corrupted Images : {len(category['corrupted'])}",

        f"Invalid Files    : {len(category['invalid'])}",

        f"Blurry Images    : {len(category['blurry'])}",

        f"Dark Images      : {len(category['dark'])}",

        f"Bright Images    : {len(category['bright'])}",

        f"Tiny Images      : {len(category['tiny'])}",

        f"Large Images     : {len(category['large'])}",

        ""

    ]

# ============================================================
# RECOMMENDATION SECTION
# ============================================================

def recommendation_section(recommendations):
    """
    Displays AI recommendations grouped by priority.
    """

    lines = [

        separator(),

        "AI RECOMMENDATIONS",

        separator()

    ]

    if not recommendations:

        lines.append("No recommendations.")
        lines.append("")

        return lines

    priority_order = [

        PRIORITY_HIGH,

        PRIORITY_MEDIUM,

        PRIORITY_LOW

    ]

    headings = {

        PRIORITY_HIGH: "HIGH PRIORITY",

        PRIORITY_MEDIUM: "MEDIUM PRIORITY",

        PRIORITY_LOW: "LOW PRIORITY"

    }

    for priority in priority_order:

        items = [

            recommendation

            for recommendation in recommendations

            if recommendation["priority"] == priority

        ]

        if not items:

            continue

        lines.append("")
        lines.append(headings[priority])
        lines.append("-" * len(headings[priority]))

        for item in items:

            lines.append(

                f"• [{item['title']}] {item['message']}"

            )

    lines.append("")

    return lines


# ============================================================
# OVERALL DATASET SECTION
# ============================================================

def overall_section(overall):
    """
    Displays overall dataset statistics.
    """

    average = overall["average_resolution"]

    minimum = overall["minimum_resolution"]

    maximum = overall["maximum_resolution"]

    lines = [

        separator(),

        "OVERALL DATASET SUMMARY",

        separator(),

        f"Fresh Images      : {overall['fresh']}",

        f"Rotten Images     : {overall['rotten']}",

        f"Total Images      : {overall['total']}",

        f"Fresh Percentage  : {overall['fresh_percentage']}%",

        f"Rotten Percentage : {overall['rotten_percentage']}%",

        "",

        "Resolution Statistics",

        f"Average Resolution : {average[0]} x {average[1]}",

        f"Minimum Resolution : {minimum[0]} x {minimum[1]}",

        f"Maximum Resolution : {maximum[0]} x {maximum[1]}",

        "",

        "Detected Issues",

        f"Duplicate Groups : {len(overall['duplicates'])}",

        f"Corrupted Images : {len(overall['corrupted'])}",

        f"Invalid Files    : {len(overall['invalid'])}",

        f"Blurry Images    : {len(overall['blurry'])}",

        f"Dark Images      : {len(overall['dark'])}",

        f"Bright Images    : {len(overall['bright'])}",

        f"Tiny Images      : {len(overall['tiny'])}",

        f"Large Images     : {len(overall['large'])}",

        ""

    ]

    return lines


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(report):
    """
    Prints the report to the console.
    """

    print(

        "\n".join(report)

    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(report):
    """
    Saves report to reports/dataset_report.txt
    """

    DATASET_REPORT.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        DATASET_REPORT,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            "\n".join(report)

        )

# ============================================================
# REPORT FOOTER
# ============================================================

def report_footer():
    """
    Creates report footer.
    """

    return [

        separator(),

        "Dataset analysis completed successfully.",

        f"Report saved to : {DATASET_REPORT}",

        separator()

    ]


# ============================================================
# GENERATE COMPLETE REPORT
# ============================================================

def generate_report(
    categories,
    overall,
    recommendations
):
    """
    Generates the complete dataset report.
    """

    report = []

    # -------------------------------------------------------
    # Header
    # -------------------------------------------------------

    report.extend(

        report_header()

    )

    # -------------------------------------------------------
    # Each Food Category
    # -------------------------------------------------------

    for category in categories:

        report.extend(

            category_report(category)

        )

        report.extend(

            resolution_section(category)

        )

        report.extend(

            format_section(category)

        )

        report.extend(

            issue_section(category)

        )

        # -------------------------------------------
        # Category Recommendations
        # -------------------------------------------

        report.extend(

            recommendation_section(

                category["recommendations"]

            )

        )

        report.append("")

    # -------------------------------------------------------
    # Overall Statistics
    # -------------------------------------------------------

    report.extend(

        overall_section(overall)

    )

    # -------------------------------------------------------
    # Overall Recommendations
    # -------------------------------------------------------

    report.extend(

        recommendation_section(

            recommendations

        )

    )

    # -------------------------------------------------------
    # Footer
    # -------------------------------------------------------

    report.extend(

        report_footer()

    )

    return report


# ============================================================
# GENERATE + PRINT + SAVE
# ============================================================

def create_report(
    categories,
    overall,
    recommendations
):
    """
    Creates, prints and saves the report.
    """

    report = generate_report(

        categories,

        overall,

        recommendations

    )

    print_report(

        report

    )

    save_report(

        report

    )

    return report
