"""Educational research reporting for RGB mold classification."""

from datetime import datetime, timezone


def generate_rgb_mold_report(prediction: dict) -> dict:
    """Create a report using actual model output and static guidance."""

    if not isinstance(prediction, dict):
        raise ValueError("Prediction must be a dictionary.")

    label = prediction.get("prediction")
    if label not in ("Mold", "No Mold"):
        raise ValueError("Unsupported mold prediction label.")

    score = float(prediction["mold_score"])
    threshold = float(prediction["threshold"])

    if not 0.0 <= score <= 1.0:
        raise ValueError("Mold score must be between 0 and 1.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if (score >= threshold) != (label == "Mold"):
        raise ValueError(
            "Prediction label and model score are inconsistent."
        )

    detected = label == "Mold"

    interpretation = (
        "The model classified the submitted image as Mold "
        "because its sigmoid output met or exceeded the "
        "configured classification threshold. This indicates "
        "similarity to the mold-positive examples learned "
        "during training. It does not confirm fungal presence, "
        "species identity, or toxin production."
        if detected else
        "The model classified the submitted image as No Mold "
        "because its sigmoid output was below the configured "
        "classification threshold. This result does not "
        "establish that the sample is free of mold, spores, "
        "mycotoxins, or other microbial contamination."
    )

    return {
        "report_type": "RGB Mold Detection Research Report",
        "report_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "analysis_overview": {
            "model": prediction.get(
                "model", "MobileMold_MobileNetV2"
            ),
            "image_type": prediction.get(
                "image_type", "Microscope-assisted RGB"
            ),
            "supported_foods": prediction.get(
                "supported_foods",
                ["tomato", "orange", "onion", "carrot"]
            ),
            "food_identification": (
                "Not determined by the mold model."
            ),
            "purpose": (
                "Experimental binary image classification "
                "for research and educational use."
            ),
        },

        "prediction_summary": {
            "predicted_class": label,
            "mold_score": score,
            "mold_score_percentage": round(score * 100, 2),
            "classification_threshold": threshold,
            "score_description": (
                "Uncalibrated sigmoid model output. "
                "This is not a validated probability "
                "that mold is present."
            ),
        },

        "scientific_interpretation": {
            "summary": interpretation,
            "scope": (
                "The prediction concerns image-level "
                "classification only. No direct fungal "
                "identification or microbiological "
                "measurement is performed."
            ),
        },

        "methodology": {
            "model_architecture": "MobileNetV2",
            "input_dimensions": "224 x 224 pixels",
            "color_channels": "RGB",
            "preprocessing": [
                "Convert the uploaded image to RGB.",
                "Resize the image to 224 x 224 pixels.",
                "Apply MobileNetV2 input normalization "
                "to the range [-1, 1].",
                "Run the trained binary classifier.",
                "Apply the configured sigmoid threshold "
                "to assign Mold or No Mold."
            ],
            "note": (
                "This describes the implemented inference "
                "pipeline, not laboratory examination."
            ),
        },

        "educational_background": {
            "title": "Understanding Fungal Growth",
            "topics": [
                {
                    "term": "Hyphae",
                    "explanation": (
                        "Thread-like structures that form "
                        "part of the vegetative growth "
                        "of many filamentous fungi."
                    ),
                },
                {
                    "term": "Mycelium",
                    "explanation": (
                        "A network of fungal hyphae."
                    ),
                },
                {
                    "term": "Spores",
                    "explanation": (
                        "Reproductive or dispersal structures "
                        "produced by many fungi."
                    ),
                },
                {
                    "term": "Mycotoxins",
                    "explanation": (
                        "Toxic substances produced by certain "
                        "fungi under particular conditions. "
                        "Their presence cannot be determined "
                        "from this model's prediction."
                    ),
                },
            ],
            "important_note": (
                "These are general educational concepts. "
                "The model does not separately detect or "
                "identify these structures or substances."
            ),
        },

        "limitations": [
            "The model is intended for microscope-assisted "
            "smartphone RGB images, not ordinary photographs.",

            "Lighting, focus, magnification, image artifacts, "
            "and sample preparation may affect predictions.",

            "The model has not been established as a "
            "general-purpose fungal species identifier.",

            "The sigmoid score is not a calibrated "
            "probability of contamination.",

            "A negative classification does not establish "
            "the absence of mold or microbial hazards.",

            "Image classification cannot establish "
            "mycotoxin presence or food safety.",

            "Performance outside the training-data "
            "distribution has not been established."
        ],

        "research_recommendations": [
            "Use images acquired under conditions comparable "
            "to those represented in model training.",

            "Document sample identity, acquisition method, "
            "magnification, and lighting separately when known.",

            "For research confirmation, consider appropriate "
            "laboratory examination by qualified personnel.",

            "Evaluate model performance using independent "
            "labeled test samples before drawing broader "
            "scientific conclusions."
        ],

        "conclusion": (
            f"The experimental MobileNetV2 model classified "
            f"the submitted image as {label}, with a raw "
            f"sigmoid mold score of {score:.4f} and a "
            f"classification threshold of {threshold:.2f}. "
            "This result is suitable for educational "
            "interpretation and exploratory image-analysis "
            "research. It is not a definitive food-safety "
            "assessment or laboratory diagnosis."
        ),

        "disclaimer": (
            "Research and educational use only. "
            "Do not use this result alone to determine "
            "whether food is safe to consume."
        ),
    }
