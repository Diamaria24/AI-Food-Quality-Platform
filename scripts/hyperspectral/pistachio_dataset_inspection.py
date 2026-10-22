import os
import re
import glob

import numpy as np


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

PISTACHIO_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "pistachio_hsi"
)


# ================================================================
# HEADER PARSER
# ================================================================

def read_hdr(hdr_path):

    metadata = {}

    try:

        with open(
            hdr_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            text = f.read()

    except Exception as e:

        return {
            "error": str(e)
        }

    # Remove line breaks around braces
    text_clean = re.sub(
        r"\{\s*([^}]*)\}",
        lambda m: "{"
        + " ".join(
            m.group(1).split()
        )
        + "}",
        text
    )

    patterns = {

        "samples":
            r"(?im)^\s*samples\s*=\s*(\d+)",

        "lines":
            r"(?im)^\s*lines\s*=\s*(\d+)",

        "bands":
            r"(?im)^\s*bands\s*=\s*(\d+)",

        "data_type":
            r"(?im)^\s*data\s*type\s*=\s*(\d+)",

        "interleave":
            r"(?im)^\s*interleave\s*=\s*([^\s]+)",

        "byte_order":
            r"(?im)^\s*byte\s*order\s*=\s*(\d+)"
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text_clean
        )

        if match:

            metadata[key] = match.group(1)

    # ------------------------------------------------------------
    # WAVELENGTH
    # ------------------------------------------------------------

    wavelength_match = re.search(
        r"(?is)wavelength\s*=\s*\{(.*?)\}",
        text_clean
    )

    if wavelength_match:

        wavelength_text = (
            wavelength_match
            .group(1)
        )

        values = []

        for value in wavelength_text.split(","):

            value = value.strip()

            try:

                values.append(
                    float(value)
                )

            except ValueError:

                pass

        metadata[
            "wavelength_count"
        ] = len(values)

        if values:

            metadata[
                "wavelength_min"
            ] = min(values)

            metadata[
                "wavelength_max"
            ] = max(values)

    return metadata


# ================================================================
# MAIN
# ================================================================

print("=" * 70)
print("PISTACHIO HSI DATASET INSPECTION")
print("=" * 70)

print(
    "\nDataset directory:"
)

print(
    PISTACHIO_DIR
)


if not os.path.isdir(
    PISTACHIO_DIR
):

    raise FileNotFoundError(
        f"\nPistachio directory not found:\n{PISTACHIO_DIR}"
    )


# ================================================================
# FIND LEVEL FOLDERS
# ================================================================

level_folders = []

for name in os.listdir(
    PISTACHIO_DIR
):

    path = os.path.join(
        PISTACHIO_DIR,
        name
    )

    if os.path.isdir(path):

        level_folders.append(
            name
        )


def level_number(name):

    match = re.search(
        r"(\d+)",
        name
    )

    if match:

        return int(
            match.group(1)
        )

    return 9999


level_folders.sort(
    key=level_number
)


print("\n" + "=" * 70)
print("LEVEL FOLDERS")
print("=" * 70)

print(
    f"\nTotal folders found: {len(level_folders)}"
)

for folder in level_folders:

    print(
        " ",
        folder
    )


# ================================================================
# EXPECTED LEVELS
# ================================================================

expected_levels = [
    f"Level {i:02d}"
    for i in range(
        1,
        27
    )
]

missing_levels = [
    level
    for level in expected_levels
    if level not in level_folders
]

extra_levels = [
    level
    for level in level_folders
    if level not in expected_levels
]

print("\n" + "=" * 70)
print("LEVEL VALIDATION")
print("=" * 70)

print(
    "\nExpected levels:",
    len(expected_levels)
)

print(
    "Found levels   :",
    len(level_folders)
)

if missing_levels:

    print(
        "\nMissing levels:"
    )

    for level in missing_levels:

        print(
            " ",
            level
        )

else:

    print(
        "\nAll Level 01–Level 26 folders found."
    )


if extra_levels:

    print(
        "\nAdditional folders:"
    )

    for level in extra_levels:

        print(
            " ",
            level
        )


# ================================================================
# SCAN FILES
# ================================================================

all_records = []

total_bil = 0
total_hdr = 0
total_png = 0

total_pairs = 0

missing_hdr = []
missing_bil = []


for level in level_folders:

    level_path = os.path.join(
        PISTACHIO_DIR,
        level
    )

    bil_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.bil"
            )
        )
    )

    hdr_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.hdr"
            )
        )
    )

    png_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.png"
            )
        )
    )

    total_bil += len(
        bil_files
    )

    total_hdr += len(
        hdr_files
    )

    total_png += len(
        png_files
    )

    bil_names = {
        os.path.splitext(
            os.path.basename(path)
        )[0]
        for path in bil_files
    }

    hdr_names = {
        os.path.splitext(
            os.path.basename(path)
        )[0]
        for path in hdr_files
    }

    common_names = (
        bil_names
        &
        hdr_names
    )

    total_pairs += len(
        common_names
    )

    for name in sorted(
        bil_names
    ):

        if name not in hdr_names:

            missing_hdr.append(
                (
                    level,
                    name
                )
            )

    for name in sorted(
        hdr_names
    ):

        if name not in bil_names:

            missing_bil.append(
                (
                    level,
                    name
                )
            )

    all_records.append({

        "level":
            level,

        "bil_count":
            len(bil_files),

        "hdr_count":
            len(hdr_files),

        "png_count":
            len(png_files),

        "bil_hdr_pairs":
            len(common_names)
    })


# ================================================================
# LEVEL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FILES PER LEVEL")
print("=" * 70)

print(
    f"\n{'Level':<12}"
    f"{'BIL':>8}"
    f"{'HDR':>8}"
    f"{'PNG':>8}"
    f"{'Pairs':>10}"
)

print(
    "-" * 50
)

for record in all_records:

    print(
        f"{record['level']:<12}"
        f"{record['bil_count']:>8}"
        f"{record['hdr_count']:>8}"
        f"{record['png_count']:>8}"
        f"{record['bil_hdr_pairs']:>10}"
    )


# ================================================================
# TOTALS
# ================================================================

print("\n" + "=" * 70)
print("TOTAL FILE COUNTS")
print("=" * 70)

print(
    "\nTotal BIL files:",
    total_bil
)

print(
    "Total HDR files:",
    total_hdr
)

print(
    "Total PNG files:",
    total_png
)

print(
    "Total BIL/HDR pairs:",
    total_pairs
)

print(
    "Missing HDR files:",
    len(missing_hdr)
)

print(
    "Missing BIL files:",
    len(missing_bil)
)


# ================================================================
# MISSING FILE PAIRS
# ================================================================

if missing_hdr:

    print("\n" + "=" * 70)
    print("BIL FILES WITHOUT HDR")
    print("=" * 70)

    for level, name in missing_hdr:

        print(
            f"{level}: {name}"
        )


if missing_bil:

    print("\n" + "=" * 70)
    print("HDR FILES WITHOUT BIL")
    print("=" * 70)

    for level, name in missing_bil:

        print(
            f"{level}: {name}"
        )


# ================================================================
# INSPECT HDR FILES
# ================================================================

print("\n" + "=" * 70)
print("HSI HEADER INSPECTION")
print("=" * 70)


header_records = []


for level in level_folders:

    level_path = os.path.join(
        PISTACHIO_DIR,
        level
    )

    hdr_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.hdr"
            )
        )
    )

    for hdr_path in hdr_files:

        metadata = read_hdr(
            hdr_path
        )

        record = {

            "level":
                level,

            "file":
                os.path.basename(
                    hdr_path
                ),

            **metadata
        }

        header_records.append(
            record
        )


# ================================================================
# HEADER DETAILS
# ================================================================

for record in header_records:

    print("\n" + "-" * 70)

    print(
        "Level:",
        record.get(
            "level"
        )
    )

    print(
        "File:",
        record.get(
            "file"
        )
    )

    print(
        "Samples:",
        record.get(
            "samples",
            "N/A"
        )
    )

    print(
        "Lines:",
        record.get(
            "lines",
            "N/A"
        )
    )

    print(
        "Bands:",
        record.get(
            "bands",
            "N/A"
        )
    )

    print(
        "Data type:",
        record.get(
            "data_type",
            "N/A"
        )
    )

    print(
        "Interleave:",
        record.get(
            "interleave",
            "N/A"
        )
    )

    print(
        "Byte order:",
        record.get(
            "byte_order",
            "N/A"
        )
    )

    print(
        "Wavelength count:",
        record.get(
            "wavelength_count",
            "N/A"
        )
    )

    print(
        "Wavelength minimum:",
        record.get(
            "wavelength_min",
            "N/A"
        )
    )

    print(
        "Wavelength maximum:",
        record.get(
            "wavelength_max",
            "N/A"
        )
    )


# ================================================================
# STRUCTURE CONSISTENCY
# ================================================================

print("\n" + "=" * 70)
print("HSI STRUCTURE CONSISTENCY")
print("=" * 70)


structure_keys = [

    "samples",

    "lines",

    "bands",

    "data_type",

    "interleave",

    "byte_order"
]


structure_groups = {}


for record in header_records:

    key = tuple(
        record.get(
            field,
            None
        )
        for field in structure_keys
    )

    structure_groups.setdefault(
        key,
        []
    ).append(
        (
            record.get("level"),
            record.get("file")
        )
    )


print(
    "\nUnique HSI structures found:",
    len(structure_groups)
)


for index, (
    structure,
    files
) in enumerate(
    structure_groups.items(),
    start=1
):

    print(
        f"\nStructure {index}:"
    )

    for field, value in zip(
        structure_keys,
        structure
    ):

        print(
            f"  {field}: {value}"
        )

    print(
        "  Number of files:",
        len(files)
    )

    if len(files) <= 10:

        for level, file in files:

            print(
                f"    {level} / {file}"
            )


# ================================================================
# WAVELENGTH CONSISTENCY
# ================================================================

wavelength_counts = set()
wavelength_mins = set()
wavelength_maxs = set()


for record in header_records:

    if (
        "wavelength_count"
        in record
    ):

        wavelength_counts.add(
            record[
                "wavelength_count"
            ]
        )

    if (
        "wavelength_min"
        in record
    ):

        wavelength_mins.add(
            record[
                "wavelength_min"
            ]
        )

    if (
        "wavelength_max"
        in record
    ):

        wavelength_maxs.add(
            record[
                "wavelength_max"
            ]
        )


print("\n" + "=" * 70)
print("WAVELENGTH CONSISTENCY")
print("=" * 70)

print(
    "\nWavelength counts:",
    sorted(
        wavelength_counts
    )
)

print(
    "Wavelength minimums:",
    sorted(
        wavelength_mins
    )
)

print(
    "Wavelength maximums:",
    sorted(
        wavelength_maxs
    )
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PISTACHIO DATASET INSPECTION SUMMARY")
print("=" * 70)

print(
    f"""
Level folders found       : {len(level_folders)}

BIL files                 : {total_bil}
HDR files                 : {total_hdr}
PNG files                 : {total_png}

BIL/HDR pairs             : {total_pairs}

Missing HDR files         : {len(missing_hdr)}
Missing BIL files         : {len(missing_bil)}

Unique HSI structures     : {len(structure_groups)}

Wavelength counts found   : {sorted(wavelength_counts)}
Wavelength minimums       : {sorted(wavelength_mins)}
Wavelength maximums       : {sorted(wavelength_maxs)}
"""
)

print("=" * 70)
print("PISTACHIO HSI INSPECTION COMPLETED")
print("=" * 70)

print(
"""
IMPORTANT:

No files were modified.

No preprocessing was performed.

No train/test split was created.

No PCA was performed.

No classifier was trained.

This was inspection only.
"""
)