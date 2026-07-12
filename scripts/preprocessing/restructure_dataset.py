"""
=========================================================
Dataset Restructuring

Author : Dia Maria
=========================================================
"""

import shutil

from pathlib import Path

from scripts.config.constants import (

    RESIZED_DATASET,
    MULTICLASS_DATASET

)

def copy_images(

    source,
    destination

):

    destination.mkdir(

        parents=True,

        exist_ok=True

    )

    count = 0

    for image in source.iterdir():

        if image.is_file():

            shutil.copy2(

                image,

                destination / image.name

            )

            count += 1

    return count

def restructure_dataset():

    print()

    print("=" * 70)

    print(

        "Restructuring Dataset..."

    )

    print("=" * 70)

    total = 0

    for split in [

        "train",
        "validation",
        "test"

    ]:

        split_path = (

            RESIZED_DATASET

            / split

        )

        print(

            f"\n{split.upper()}"

        )

        for food in sorted(

            split_path.iterdir()

        ):

            if not food.is_dir():

                continue

            for quality in [

                "Fresh",
                "Rotten"

            ]:

                folder = (

                    food
                    / quality

                )

                if not folder.exists():

                    continue

                class_name = (

                    f"{food.name.capitalize()}"

                    f"_{quality}"

                )

                destination = (

                    MULTICLASS_DATASET

                    / split

                    / class_name

                )

                count = copy_images(

                    folder,

                    destination

                )

                total += count

                print(

                    f"[OK] "

                    f"{class_name:<25}"

                    f"{count}"

                )

    print()

    print("=" * 70)

    print(

        f"Total Images : {total}"

    )

    print("=" * 70)

if __name__ == "__main__":

    restructure_dataset()


