"""
Rule-based consumer guidance for RGB freshness results.

This guidance is informational. It is not an additional AI
prediction and cannot certify food safety or remaining shelf life.
"""

FOOD_GUIDANCE = {
    "Apple": {
        "storage": [
            "Refrigerate whole apples to help maintain freshness.",
            "Keep apples dry and inspect them periodically.",
            "Store cut apples in a covered container in the refrigerator."
        ],
        "spoilage_signs": [
            "Extensive soft or mushy areas",
            "Visible mold or decaying patches",
            "Leaking liquid or an unusual odor"
        ]
    },
    "Banana": {
        "storage": [
            "Keep unripe bananas at room temperature until ripe.",
            "Refrigerate ripe bananas to slow further ripening; the peel may darken.",
            "Store cut bananas in a covered container in the refrigerator."
        ],
        "spoilage_signs": [
            "Leaking or extensively mushy flesh",
            "Visible mold",
            "An unusual fermented or rotten odor"
        ]
    },
    "Strawberry": {
        "storage": [
            "Refrigerate strawberries promptly.",
            "Keep them dry and wash just before eating.",
            "Remove visibly spoiled berries and inspect nearby berries."
        ],
        "spoilage_signs": [
            "Visible fuzzy mold",
            "Soft, leaking or collapsing berries",
            "Unusual odor or extensive decay"
        ]
    },
    "Tomato": {
        "storage": [
            "Allow unripe tomatoes to ripen at room temperature.",
            "Refrigerate ripe tomatoes if they will not be eaten promptly.",
            "Refrigerate cut tomatoes in a covered container."
        ],
        "spoilage_signs": [
            "Visible mold",
            "Leaking or extensively mushy areas",
            "Unusual odor or extensive decay"
        ]
    }
}


def get_food_guidance(food, freshness):
    """Return general storage and inspection guidance."""

    details = FOOD_GUIDANCE.get(food)

    if details is None:
        return {
            "available": False,
            "message": "No fruit-specific guidance is available."
        }

    is_fresh = str(freshness).lower() == "fresh"
    is_rotten = str(freshness).lower() == "rotten"

    if is_fresh:
        recommendation = (
            "The image was classified as Fresh. Follow the "
            "storage guidance and inspect the actual food "
            "before consumption. This result does not "
            "establish that the food is safe to eat."
        )
    elif is_rotten:
        recommendation = (
            "The image was classified as Rotten. Inspect "
            "the food for visible deterioration. Discard "
            "food showing mold or significant decay. "
            "Do not taste food to determine whether it "
            "is safe."
        )
    else:
        recommendation = (
            "Inspect the food and follow appropriate "
            "storage and food-safety practices."
        )

    return {
        "available": True,
        "food": food,
        "classification": freshness,
        "recommendation": recommendation,
        "storage_tips": details["storage"],
        "visible_spoilage_signs": details["spoilage_signs"],
        "guidance_type": "rule_based",
        "disclaimer": (
            "General storage and spoilage guidance only. "
            "The listed signs were not individually detected "
            "by the image model. No remaining shelf life "
            "or food-safety certification is provided."
        )
    }
