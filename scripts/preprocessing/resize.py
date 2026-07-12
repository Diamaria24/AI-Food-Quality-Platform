"""
=========================================================
Dataset Resizer
=========================================================
"""

from pathlib import Path

from PIL import Image

from scripts.config.constants import (

    PROCESSED_DATASET,
    RESIZED_DATASET,
    IMAGE_SIZE,
    IMAGE_EXTENSIONS

)

def resize_image(

    image_path,
    output_path

):

    try:

        with Image.open(

            image_path

        ) as image:

            image = image.convert(

                "RGB"

            )

            image = image.resize(

                IMAGE_SIZE

            )

            output_path.parent.mkdir(

                parents=True,
                exist_ok=True

            )

            image.save(

                output_path,

                format="PNG"

            )

    except Exception as error:

        print(

            f"[ERROR] {image_path}"

        )

        print(error)

def resize_dataset():

    print()

    print("=" * 70)

    print("Resizing Dataset...")

    print("=" * 70)

    count = 0

    for split in [

        "train",
        "validation",
        "test"

    ]:

        split_path = (

            PROCESSED_DATASET

            / split

        )

        for file in split_path.rglob("*"):

            if (

                file.is_file()

                and

                file.suffix.lower()

                in IMAGE_EXTENSIONS

            ):

                relative = (

                    file.relative_to(

                        PROCESSED_DATASET

                    )

                )

                output = (

                    RESIZED_DATASET

                    / relative

                )

                output = output.with_suffix(

                    ".png"

                )

                resize_image(

                    file,

                    output

                )

                count += 1

                if count % 1000 == 0:

                    print(

                        f"{count} images..."
                    )

    print()

    print(

        f"Total Images: {count}"

    )

if __name__ == "__main__":

    resize_dataset()

