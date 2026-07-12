"""
=========================================================
AI Food Quality Platform

Dataset Splitter

Author : Dia Maria
=========================================================
"""

import shutil

from pathlib import Path

from sklearn.model_selection import train_test_split

from scripts.config.constants import (
    RGB_DATASET,
    PROCESSED_DATASET,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    IMAGE_EXTENSIONS
)

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

def create_output_folders():

    for split in [

        "train",
        "validation",
        "test"

    ]:

        (PROCESSED_DATASET / split).mkdir(

            parents=True,
            exist_ok=True

        )

# =========================================================
# GET IMAGE FILES
# =========================================================

def get_images(folder):

    images = [

        file

        for file in folder.iterdir()

        if (

            file.is_file()

            and

            file.suffix.lower()

            in IMAGE_EXTENSIONS

        )

    ]

    return images

# =========================================================
# COPY FILES
# =========================================================

def copy_images(

    files,
    destination

):

    destination.mkdir(

        parents=True,
        exist_ok=True

    )

    for file in files:

        shutil.copy2(

            file,
            destination / file.name

        )

# =========================================================
# SPLIT CLASS
# =========================================================

def split_class(

    source_folder,
    food,
    category

):

    images = get_images(

        source_folder

    )

    if not images:

        return

    # -------------------------------------
    # TRAIN
    # -------------------------------------

    train_files, temp_files = train_test_split(

        images,

        train_size=TRAIN_SPLIT,

        random_state=42,

        shuffle=True

    )

    # -------------------------------------
    # VALIDATION + TEST
    # -------------------------------------

    validation_ratio = (

        VALIDATION_SPLIT /

        (

            VALIDATION_SPLIT

            +

            TEST_SPLIT

        )

    )

    validation_files, test_files = (

        train_test_split(

            temp_files,

            train_size=validation_ratio,

            random_state=42,

            shuffle=True

        )

    )

    # -------------------------------------
    # Copy
    # -------------------------------------

    copy_images(

        train_files,

        PROCESSED_DATASET
        / "train"
        / food
        / category

    )

    copy_images(

        validation_files,

        PROCESSED_DATASET
        / "validation"
        / food
        / category

    )

    copy_images(

        test_files,

        PROCESSED_DATASET
        / "test"
        / food
        / category

    )

    print(

        f"[OK] {food} - {category}"

    )

# =========================================================
# SPLIT DATASET
# =========================================================

def split_dataset():

    print()

    print("=" * 70)

    print(

        "Splitting Dataset..."

    )

    print("=" * 70)

    create_output_folders()

    for food in sorted(

        RGB_DATASET.iterdir()

    ):

        if not food.is_dir():

            continue

        print(

            f"\n{food.name}"

        )

        for category in [

            "Fresh",
            "Rotten"

        ]:

            folder = (

                food
                / category

            )

            if folder.exists():

                split_class(

                    folder,

                    food.name,

                    category

                )

    print()

    print("=" * 70)

    print(

        "Dataset Split Completed"

    )

    print("=" * 70)

if __name__ == "__main__":

    split_dataset()

