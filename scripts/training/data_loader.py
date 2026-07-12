"""
=========================================================
Data Loader
=========================================================
"""

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

from scripts.config.constants import (

    MULTICLASS_TRAIN,
    MULTICLASS_VALIDATION,
    MULTICLASS_TEST,

    IMAGE_SIZE,

    BATCH_SIZE

)

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    brightness_range=(0.8, 1.2),

    horizontal_flip=True,

    fill_mode="nearest"

)

test_datagen = ImageDataGenerator(

    rescale=1./255

)

def load_datasets():

    train_generator = (

        train_datagen.flow_from_directory(

            MULTICLASS_TRAIN,

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            shuffle=True,

            class_mode="categorical"

        )

    )

    validation_generator = (

        test_datagen.flow_from_directory(

            MULTICLASS_VALIDATION,

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            class_mode="categorical"

        )

    )

    test_generator = (

        test_datagen.flow_from_directory(

            MULTICLASS_TEST,

            target_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            class_mode="categorical",

            shuffle=False

        )

    )

    return (

        train_generator,

        validation_generator,

        test_generator

    )

if __name__ == "__main__":

    train_gen, val_gen, test_gen = (

        load_datasets()

    )

    print()

    print("Classes")

    print(

        train_gen.class_indices

    )

    print()

    print(

        f"Train Images : {train_gen.samples}"

    )

    print(

        f"Validation Images : {val_gen.samples}"

    )

    print(

        f"Test Images : {test_gen.samples}"

    )


