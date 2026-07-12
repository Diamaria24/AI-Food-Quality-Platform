"""
=========================================================
Train MobileNetV2
=========================================================
"""

import json

from tensorflow.keras.callbacks import (

    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint

)

from scripts.training.mobilenet_loader import (
    load_datasets
)

from scripts.training.mobilenet_model import (
    build_mobilenet
)

from scripts.config.constants import (

    TRAINED_MODELS,
    HISTORY_FOLDER

)

print()
print("="*70)
print("Loading Dataset...")
print("="*70)

train_data, val_data, test_data = (
    load_datasets()
)

print()
print("="*70)
print("Building MobileNet...")
print("="*70)

model = build_mobilenet()

model.summary()

callbacks = [

    EarlyStopping(

        monitor="val_accuracy",

        patience=5,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        verbose=1

    ),

    ModelCheckpoint(

        TRAINED_MODELS /
        "mobilenet.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    )
]

print()
print("="*70)
print("Training Started...")
print("="*70)

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=15,

    callbacks=callbacks

)

history_file = (

    HISTORY_FOLDER /
    "mobilenet_history.json"

)

with open(

    history_file,

    "w"

) as f:

    json.dump(

        history.history,

        f,

        indent=4

    )

print()
print("="*70)
print("Training Completed")
print("="*70)

print(

    f"Model Saved : "

    f"{TRAINED_MODELS}"

)

print(

    f"History Saved : "

    f"{history_file}"

)

