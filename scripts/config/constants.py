"""
=============================================================
AI Food Quality Platform

Configuration Constants

Author : Dia Maria
=============================================================
"""

from pathlib import Path

# =============================================================
# PROJECT PATHS
# =============================================================

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset Paths
DATASET_ROOT = PROJECT_ROOT / "datasets"

RAW_DATASET = DATASET_ROOT / "raw"

RGB_DATASET = RAW_DATASET / "rgb"

HYPERSPECTRAL_DATASET = RAW_DATASET / "hyperspectral"

PROCESSED_DATASET = DATASET_ROOT / "processed"

TRAIN_DATASET = DATASET_ROOT / "train"

VALIDATION_DATASET = DATASET_ROOT / "validation"

TEST_DATASET = DATASET_ROOT / "test"

# Reports
REPORTS_FOLDER = PROJECT_ROOT / "reports"

DATASET_REPORT = REPORTS_FOLDER / "dataset_report.txt"

PREPROCESSING_REPORT = REPORTS_FOLDER / "preprocessing_report.txt"

MODEL_REPORT = REPORTS_FOLDER / "model_report.txt"

# Models
MODELS_FOLDER = PROJECT_ROOT / "models"

SAVED_MODELS = MODELS_FOLDER / "saved_models"

# Trained Models
TRAINED_MODELS = (

    MODELS_FOLDER

    / "trained"

)

# Output
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

GRAPH_FOLDER = OUTPUT_FOLDER / "graphs"

LOG_FOLDER = OUTPUT_FOLDER / "logs"

PREDICTION_FOLDER = OUTPUT_FOLDER / "predictions"

# Training Outputs
HISTORY_FOLDER = (

    OUTPUT_FOLDER

    / "history"

)

PLOTS_FOLDER = (

    OUTPUT_FOLDER

    / "plots"

)


# =============================================================
# IMAGE FILES
# =============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# =============================================================
# IMAGE QUALITY THRESHOLDS
# =============================================================

# Variance of Laplacian
BLUR_THRESHOLD = 100

# Brightness (0 - 255)
DARK_THRESHOLD = 50

BRIGHT_THRESHOLD = 205

# Tiny Images
MIN_WIDTH = 100

MIN_HEIGHT = 100

# Extremely Large Images
MAX_WIDTH = 3000

MAX_HEIGHT = 3000


# =============================================================
# PREPROCESSING
# =============================================================

# Recommended resize for CNN models
IMAGE_SIZE = (224,224)

# RGB Channels
CHANNELS = 3

# Normalization
PIXEL_MIN = 0

PIXEL_MAX = 255


# =============================================================
# DATASET HEALTH SCORE
# =============================================================

INITIAL_HEALTH_SCORE = 100

HIGHLY_IMBALANCED_PENALTY = 10

MODERATE_IMBALANCE_PENALTY = 5

CORRUPTED_IMAGE_PENALTY = 3

DUPLICATE_IMAGE_PENALTY = 2

BLURRY_IMAGE_PENALTY = 2

INVALID_FILE_PENALTY = 1

TINY_IMAGE_PENALTY = 1


# =============================================================
# REPORT
# =============================================================

LINE_LENGTH = 90

DATE_FORMAT = "%d-%m-%Y %H:%M:%S"


# =============================================================
# MACHINE LEARNING
# =============================================================

TRAIN_SPLIT = 0.70

VALIDATION_SPLIT = 0.15

TEST_SPLIT = 0.15

RANDOM_STATE = 42


# =============================================================
# SUPPORTED IMAGE FORMATS
# =============================================================

SUPPORTED_FORMATS = (
    "JPEG",
    "PNG",
    "BMP",
    "TIFF"
)


# =============================================================
# STATUS LABELS
# =============================================================

STATUS_EXCELLENT = "Excellent"

STATUS_GOOD = "Good"

STATUS_MODERATE = "Moderate"

STATUS_HIGHLY_IMBALANCED = "Highly Imbalanced"

STATUS_INVALID = "Invalid"


# =============================================================
# DATASET RECOMMENDATIONS
# =============================================================

RECOMMENDED_RESIZE = (224, 224)

MIN_IMAGES_PER_CLASS = 200

# =============================================================
# PRIORITY LEVELS
# =============================================================

PRIORITY_HIGH = "HIGH"

PRIORITY_MEDIUM = "MEDIUM"

PRIORITY_LOW = "LOW"
# =============================================================
# CREATE REQUIRED DIRECTORIES
# =============================================================

REQUIRED_DIRECTORIES = [

    REPORTS_FOLDER,

    OUTPUT_FOLDER,

    GRAPH_FOLDER,

    LOG_FOLDER,

    PREDICTION_FOLDER,

    HISTORY_FOLDER,

    PLOTS_FOLDER,

    MODELS_FOLDER,

    SAVED_MODELS,

    TRAINED_MODELS

]


RESIZED_DATASET = (

    DATASET_ROOT

    / "resized"

)

MULTICLASS_DATASET = (

    DATASET_ROOT

    / "multiclass"

)

# =============================================================
# MULTICLASS DATASET
# =============================================================

MULTICLASS_DATASET = (

    DATASET_ROOT

    / "multiclass"

)

MULTICLASS_TRAIN = (

    MULTICLASS_DATASET

    / "train"

)

MULTICLASS_VALIDATION = (

    MULTICLASS_DATASET

    / "validation"

)

MULTICLASS_TEST = (

    MULTICLASS_DATASET

    / "test"

)


# =============================================================
# TRAINING
# =============================================================

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.001

MODEL_FOLDER = (

    PROJECT_ROOT

    / "models"

)

TRAINED_MODELS = (

    MODEL_FOLDER

    / "trained"

)

GRAPHS_FOLDER = (

    OUTPUT_FOLDER

    / "graphs"

)

EPOCHS = 20

PATIENCE = 5

MODEL_NAME = "baseline_cnn.keras"


for directory in REQUIRED_DIRECTORIES:

    directory.mkdir(

        parents=True,

        exist_ok=True

    )

    