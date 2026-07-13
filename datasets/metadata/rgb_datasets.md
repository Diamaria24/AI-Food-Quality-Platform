# RGB Datasets Used

The RGB module was developed using publicly available fruit quality datasets.

## Dataset Sources

### 1. Fruit Freshness Dataset
Source: Kaggle

Description:
Contains fresh and rotten fruit images for classification.

Classes:
- Apple Fresh
- Apple Rotten
- Banana Fresh
- Banana Rotten
- Strawberry Fresh
- Strawberry Rotten
- Tomato Fresh
- Tomato Rotten

Total Images Used:
14,557 RGB images

Purpose:
Used for training, validation, and testing of CNN and MobileNetV2 models.

---

### Final Dataset Distribution

| Class | Images |
|--------|---------|
| Apple Fresh | 2438 |
| Apple Rotten | 2925 |
| Banana Fresh | 1999 |
| Banana Rotten | 2797 |
| Strawberry Fresh | 1603 |
| Strawberry Rotten | 1596 |
| Tomato Fresh | 604 |
| Tomato Rotten | 595 |

---

### Dataset Split

| Split | Images | Percentage |
|-------|---------|------------|
| Training | 9,934 | 68.24% |
| Validation | 2,176 | 14.95% |
| Testing | 2,174 | 14.93% |

---

### Note

The original datasets are not included in this repository due to storage limitations.

Please download the datasets from the respective sources (kaggle) and place them inside:

datasets/raw/
datasets/processed/
