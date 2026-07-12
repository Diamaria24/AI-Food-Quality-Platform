"""
=========================================================
MobileNetV2 Transfer Learning Model
=========================================================
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (

    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization

)

from tensorflow.keras.optimizers import Adam

from scripts.config.constants import (
    IMAGE_SIZE
)


def build_mobilenet():

    base_model = MobileNetV2(

        weights="imagenet",

        include_top=False,

        input_shape=(

            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )

    )

    # Freeze feature extractor

    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(

        256,

        activation="relu"

    )(x)

    x = Dropout(0.50)(x)

    outputs = Dense(

        8,

        activation="softmax"

    )(x)

    model = Model(

        inputs=base_model.input,

        outputs=outputs

    )

    model.compile(

        optimizer=Adam(
            learning_rate=1e-4
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


if __name__ == "__main__":

    model = build_mobilenet()

    model.summary()


