import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)

# ==========================================
# Choose any good sample image
# ==========================================

IMAGE_PATH = (
    r"datasets\multiclass\train\Apple_Fresh"
)

# Automatically pick first image
image_file = None

for f in os.listdir(IMAGE_PATH):

    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        image_file = os.path.join(
            IMAGE_PATH,
            f
        )
        break


SAVE_DIR = (
    r"outputs\plots\augmentations"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

print("Using image:")
print(image_file)

# ==========================================
# Load image
# ==========================================

img = load_img(
    image_file,
    target_size=(224, 224)
)

x = img_to_array(img)

x = np.expand_dims(
    x,
    axis=0
)

# ==========================================
# Augmentations
# ==========================================

datagen = ImageDataGenerator(

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    shear_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.7, 1.3],

    fill_mode="nearest"
)

# ==========================================
# Generate images
# ==========================================

count = 0

for batch in datagen.flow(

        x,

        batch_size=1,

        save_to_dir=SAVE_DIR,

        save_prefix="aug",

        save_format="jpg"
):

    count += 1

    if count >= 8:
        break

print()
print("Saved augmentation images in:")
print(SAVE_DIR)
