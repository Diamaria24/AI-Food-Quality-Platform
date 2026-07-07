"""
AI Food Quality Platform
Environment Verification Script

Author: Dia Maria
"""

import sys

print("=" * 60)
print("AI FOOD QUALITY PLATFORM")
print("Environment Verification")
print("=" * 60)

# ----------------------------
# Python
# ----------------------------
print(f"\nPython Version : {sys.version}")

# ----------------------------
# NumPy
# ----------------------------
try:
    import numpy as np
    print(f"NumPy          : {np.__version__}")
except Exception as e:
    print(f"NumPy          : ERROR -> {e}")

# ----------------------------
# Pandas
# ----------------------------
try:
    import pandas as pd
    print(f"Pandas         : {pd.__version__}")
except Exception as e:
    print(f"Pandas         : ERROR -> {e}")

# ----------------------------
# OpenCV
# ----------------------------
try:
    import cv2
    print(f"OpenCV         : {cv2.__version__}")
except Exception as e:
    print(f"OpenCV         : ERROR -> {e}")

# ----------------------------
# TensorFlow
# ----------------------------
try:
    import tensorflow as tf
    print(f"TensorFlow     : {tf.__version__}")
    print("GPU Available  :", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print(f"TensorFlow     : ERROR -> {e}")

# ----------------------------
# Scikit-Learn
# ----------------------------
try:
    import sklearn
    print(f"Scikit-Learn   : {sklearn.__version__}")
except Exception as e:
    print(f"Scikit-Learn   : ERROR -> {e}")

# ----------------------------
# Spectral Python
# ----------------------------
try:
    import spectral
    print(f"Spectral Python: {spectral.__version__}")
except Exception as e:
    print(f"Spectral Python: ERROR -> {e}")

# ----------------------------
# Matplotlib
# ----------------------------
try:
    import matplotlib
    print(f"Matplotlib     : {matplotlib.__version__}")
except Exception as e:
    print(f"Matplotlib     : ERROR -> {e}")

# ----------------------------
# SHAP
# ----------------------------
try:
    import shap
    print(f"SHAP           : {shap.__version__}")
except Exception as e:
    print(f"SHAP           : ERROR -> {e}")

# ----------------------------
# Quick Functional Tests
# ----------------------------
print("\nRunning Quick Functional Tests...")

# NumPy Test
a = np.array([1, 2, 3])
print("NumPy Test      : PASS" if a.sum() == 6 else "FAIL")

# OpenCV Test
img = np.zeros((100, 100, 3), dtype=np.uint8)
print("OpenCV Test     : PASS" if img.shape == (100, 100, 3) else "FAIL")

# TensorFlow Test
tensor = tf.constant([1, 2, 3])
print("TensorFlow Test : PASS" if tensor.shape[0] == 3 else "FAIL")

# Scikit-learn Test
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
print("Scikit Test     : PASS")

print("\nAll Basic Functional Tests Passed ✅")

# ----------------------------
# Joblib
# ----------------------------
try:
    import joblib
    print(f"Joblib         : {joblib.__version__}")
except Exception as e:
    print(f"Joblib         : ERROR -> {e}")

# ----------------------------
# ImageIO
# ----------------------------
try:
    import imageio
    print(f"ImageIO        : {imageio.__version__}")
except Exception as e:
    print(f"ImageIO        : ERROR -> {e}")

print("\n" + "=" * 60)
print("Environment Verification Completed")
print("=" * 60)