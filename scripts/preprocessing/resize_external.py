from pathlib import Path
from PIL import Image

SIZE = (128,128)

SOURCE = Path(
    "datasets/raw/external_test_restructured"
)

DESTINATION = Path(
    "datasets/resized_external"
)

DESTINATION.mkdir(
    parents=True,
    exist_ok=True
)

count = 0

for folder in SOURCE.iterdir():

    if not folder.is_dir():
        continue

    output = DESTINATION / folder.name
    output.mkdir(exist_ok=True)

    for image_path in folder.glob("*"):

        try:

            image = Image.open(image_path)

            image = image.convert("RGB")

            image = image.resize(SIZE)

            image.save(

                output / image_path.name

            )

            count += 1

        except:

            pass

print()

print("Images Resized :", count)