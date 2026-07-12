"""
=========================================================
Baseline CNN Model
=========================================================
"""

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Input,

    Conv2D,
    MaxPooling2D,

    GlobalAveragePooling2D,

    Dense,
    Dropout,

    BatchNormalization

)

from tensorflow.keras.optimizers import Adam

from scripts.config.constants import (

    IMAGE_SIZE,
    LEARNING_RATE

)

# =========================================================
# BUILD CNN MODEL
# =========================================================

def build_baseline_cnn():

    model = Sequential()

    model.add(

        Input(

            shape=(

                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3

            )

        )

    )

    model.add(

        Conv2D(

            32,

            (3, 3),

            activation="relu",

            padding="same"

        )

    )

    model.add(

        BatchNormalization()

    )

    model.add(

        MaxPooling2D(

            (2, 2)

        )

    )

    model.add(

        Conv2D(

            64,

            (3, 3),

            activation="relu",

            padding="same"

        )

    )

    model.add(

        BatchNormalization()

    )

    model.add(

        MaxPooling2D(

            (2, 2)

        )

    )

    model.add(

        Conv2D(

            128,

            (3, 3),

            activation="relu",

            padding="same"

        )

    )

    model.add(

        BatchNormalization()

    )

    model.add(

        MaxPooling2D(

            (2, 2)

        )

    )

    model.add(

        GlobalAveragePooling2D()

    )

    model.add(

        Dense(

            128,

            activation="relu"

        )

    )

    model.add(

        Dropout(

            0.50

        )

    )

    model.add(

        Dense(

            8,

            activation="softmax"

        )

    )

    model.compile(

        optimizer=Adam(

            learning_rate=LEARNING_RATE

        ),

        loss="categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )

    return model

if __name__ == "__main__":

    model = (

        build_baseline_cnn()

    )

    model.summary()

