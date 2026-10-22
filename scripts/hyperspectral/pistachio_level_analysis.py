import os
import re
import glob
from PIL import Image, ImageDraw, ImageFont


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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "pistachio_hsi"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# PRINT HEADER INFORMATION
# ================================================================

def inspect_hdr(hdr_path):

    print("\n" + "=" * 70)
    print("HDR FILE")
    print("=" * 70)

    print(
        os.path.basename(hdr_path)
    )

    try:

        with open(
            hdr_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            lines = f.readlines()

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        return

    print(
        "\nFULL HEADER CONTENT:"
    )

    print(
        "-" * 70
    )

    for line in lines:

        line = line.rstrip()

        if line:

            print(line)


# ================================================================
# FIND LEVELS
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

        match = re.search(
            r"(\d+)",
            name
        )

        if match:

            level_folders.append(
                (
                    int(match.group(1)),
                    name
                )
            )


level_folders.sort()


print("=" * 70)
print("PISTACHIO LEVEL ANALYSIS")
print("=" * 70)

print(
    f"\nTotal levels found: {len(level_folders)}"
)


# ================================================================
# PNG INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("PNG IMAGE INFORMATION")
print("=" * 70)


image_records = []


for number, level in level_folders:

    level_path = os.path.join(
        PISTACHIO_DIR,
        level
    )

    png_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.png"
            )
        )
    )

    print(
        f"\n{level}"
    )

    print(
        "-" * 50
    )

    for png_path in png_files:

        try:

            with Image.open(
                png_path
            ) as img:

                width, height = img.size

                mode = img.mode

                print(
                    f"{os.path.basename(png_path)}"
                )

                print(
                    f"  Size : {width} x {height}"
                )

                print(
                    f"  Mode : {mode}"
                )

                image_records.append(
                    {
                        "level": level,
                        "number": number,
                        "path": png_path,
                        "width": width,
                        "height": height
                    }
                )

        except Exception as e:

            print(
                f"  ERROR reading PNG: {e}"
            )


# ================================================================
# CREATE CONTACT SHEET
# ================================================================

print("\n" + "=" * 70)
print("CREATING PNG CONTACT SHEET")
print("=" * 70)


THUMB_WIDTH = 300
THUMB_HEIGHT = 230

LABEL_HEIGHT = 45

COLUMNS = 4

ROWS = (
    len(image_records)
    + COLUMNS
    - 1
) // COLUMNS


sheet_width = (
    COLUMNS
    * THUMB_WIDTH
)

sheet_height = (
    ROWS
    * (
        THUMB_HEIGHT
        + LABEL_HEIGHT
    )
)


contact_sheet = Image.new(
    "RGB",
    (
        sheet_width,
        sheet_height
    ),
    "white"
)


draw = ImageDraw.Draw(
    contact_sheet
)


for index, record in enumerate(
    image_records
):

    try:

        img = Image.open(
            record["path"]
        ).convert(
            "RGB"
        )

        img.thumbnail(
            (
                THUMB_WIDTH - 20,
                THUMB_HEIGHT - 20
            )
        )

        x = (
            index
            % COLUMNS
        ) * THUMB_WIDTH

        y = (
            index
            // COLUMNS
        ) * (
            THUMB_HEIGHT
            + LABEL_HEIGHT
        )

        image_x = (
            x
            + (
                THUMB_WIDTH
                - img.width
            ) // 2
        )

        image_y = (
            y
            + (
                THUMB_HEIGHT
                - img.height
            ) // 2
        )

        contact_sheet.paste(
            img,
            (
                image_x,
                image_y
            )
        )

        label = (
            f"{record['level']}  |  "
            f"{os.path.basename(record['path'])}"
        )

        draw.text(
            (
                x + 10,
                y + THUMB_HEIGHT
            ),
            label,
            fill="black"
        )

    except Exception as e:

        print(
            "Contact sheet error:",
            e
        )


contact_sheet_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_levels_contact_sheet.png"
)


contact_sheet.save(
    contact_sheet_path
)


print(
    "\nContact sheet saved:"
)

print(
    contact_sheet_path
)


# ================================================================
# HDR SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("HDR METADATA SUMMARY")
print("=" * 70)


all_hdr_files = []


for number, level in level_folders:

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

        all_hdr_files.append(
            (
                level,
                hdr_path
            )
        )


print(
    f"\nTotal HDR files: {len(all_hdr_files)}"
)


# ================================================================
# DISPLAY FIRST HDR FILES IN DETAIL
# ================================================================

print("\n" + "=" * 70)
print("DETAILED HDR INSPECTION")
print("=" * 70)

print(
"""
The first two HDR files will be printed in full.

We are specifically looking for fields such as:

description
sample name
class
level
category
acquisition information
sensor information
wavelength
gain
offset
"""
)


for index, (
    level,
    hdr_path
) in enumerate(
    all_hdr_files
):

    print(
        f"\n\nLEVEL: {level}"
    )

    inspect_hdr(
        hdr_path
    )

    if index >= 1:

        break


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PISTACHIO LEVEL ANALYSIS COMPLETED")
print("=" * 70)

print(
f"""
Levels found        : {len(level_folders)}
PNG images found    : {len(image_records)}
HDR files found     : {len(all_hdr_files)}

Contact sheet:
{contact_sheet_path}

IMPORTANT:

No BIL files were loaded.

No HSI data was modified.

No preprocessing was performed.

No train/test split was created.

No PCA was performed.

No classifier was trained.
"""
)

print("=" * 70)