🍎 AI Food Quality Platform

An AI-powered food quality assessment platform that utilizes Deep Learning, Computer Vision, and Hyperspectral Imaging concepts to perform freshness estimation, spoilage detection, and early mold identification.

📖 Project Overview

Food spoilage contributes significantly to global food waste and economic losses. This project aims to develop an intelligent and explainable AI system capable of automatically evaluating food quality using image-based analysis.

The platform currently focuses on RGB image analysis and is designed for future integration with hyperspectral sensors for early-stage mold detection.

🎯 Objectives

Automatic food freshness assessment
Detection of defective and spoiled produce
Real-time prediction through a web application
Explainable AI using GradCAM visualizations
Future support for hyperspectral imaging and low-cost sensors

🚀 Features

RGB Analysis Module

✅ Multi-class Fruit Classification

✅ Fresh vs Rotten Detection

✅ Confidence Score Prediction

✅ Real-Time Web Deployment

✅ Explainable AI (GradCAM)

✅ External Dataset Validation

Future Hyperspectral Module

🔄 Spectral Data Processing

🔄 PCA-based Dimensionality Reduction

🔄 Spectral Signature Visualization

🔄 Early Mold Detection

🔄 Sensor Integration (AS7341)

🧠 Supported Classes

The RGB module currently supports 8 classes:

Apple Fresh
Apple Rotten
Banana Fresh
Banana Rotten
Strawberry Fresh
Strawberry Rotten
Tomato Fresh
Tomato Rotten

📊 Dataset Information

Final RGB Dataset

Class	Images
Apple Fresh	2438
Apple Rotten	2925
Banana Fresh	1999
Banana Rotten	2797
Strawberry Fresh	1603
Strawberry Rotten	1596
Tomato Fresh	604
Tomato Rotten	595
Total Images
14,557 Images
Dataset Split
Set	Images	Percentage
Training	9,934	69.55%
Validation	2,176	15.23%
Testing	2,174	15.22%

🏗 System Architecture

Input Image
      ↓
Preprocessing
      ↓
CNN / MobileNetV2
      ↓
Prediction
      ↓
GradCAM Explainability
      ↓
Streamlit Deployment

🛠 Tech Stack

Programming
Python 3.11
Deep Learning
TensorFlow
Keras
Scikit-Learn
Computer Vision
OpenCV
Pillow
Hyperspectral Imaging
Spectral Python (SPy)
Visualization
Matplotlib
Plotly
Seaborn
Explainable AI
GradCAM
SHAP (Planned)
Web Deployment
Streamlit

📂 Project Structure

AI-Food-Quality-Platform/

├── app.py
├── datasets/
├── models/
│   └── trained/
├── outputs/
│   ├── plots/
│   ├── predictions/
│   └── reports/
├── scripts/
│   ├── training/
│   ├── preprocessing/
│   └── evaluation/
├── notebooks/
├── reports/
├── docs/
├── tests/
├── configs/
└── requirements.txt

📈 Model Performance

Baseline CNN
Metric	Value
Test Accuracy	96.78%
Weighted F1	0.97
External Accuracy	88.06%
MobileNetV2 (Final Model)
Metric	Value
Test Accuracy	99.17%
Validation Accuracy	98.76%
Test Loss	0.0291
Weighted F1	0.99
External Accuracy	88.71%

📌 Key Findings

✅ MobileNetV2 outperformed the custom CNN architecture.

✅ Transfer learning improved performance by approximately 2.39%.

✅ Near-perfect ROC-AUC values (0.999–1.000) were achieved.

✅ External testing revealed domain shift, particularly for tomato classes.

🔥 Explainable AI

The project incorporates GradCAM visualizations to interpret model decisions.

Features:

Heatmap generation
Identification of defective regions
Increased model transparency
Better trust and interpretability

🌐 Web Application

The platform includes a real-time Streamlit application that allows users to:

✅ Upload fruit images

✅ Obtain freshness predictions

✅ View confidence scores

✅ Visualize GradCAM explanations

📷 Sample Outputs

Confusion Matrix
ROC Curves
Accuracy/Loss Curves
Sample Predictions
GradCAM Heatmaps
Streamlit Interface

🚀 Future Work

Hyperspectral imaging integration
AS7341 sensor integration
Shelf-life prediction
Recommendation engine
Cloud deployment
Mobile application
Edge AI deployment

📌 Current Development Status

Module	Status
Research & Planning	✅
Dataset Preparation	✅
Exploratory Data Analysis	✅
Baseline CNN	✅
MobileNetV2	✅
Explainable AI	✅
Streamlit Deployment	✅
External Validation	✅
Hyperspectral Module	🔄
Hardware Integration	🔄

👩‍💻 Authors

Dia Maria
Fidha Shaji 
SreeLakshmi 
Nayana James

📜 License

This project is developed for:

Academic Research
Educational Purposes
Software Engineering Project Work

⭐ Citation

If you use this project in your research:

@project{AIFoodQualityPlatform,
  title={AI Food Quality Platform using Deep Learning and Hyperspectral Imaging},
  author={Dia Maria et al.},
  year={2026}
}

## Screenshots
1. Architecture Diagram
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/93cc852f-b11b-4b73-aba3-bf18f05e5cbf" />


![Streamlit](outputs/screenshots/app.png)

This makes the GitHub repository look much more professional.
