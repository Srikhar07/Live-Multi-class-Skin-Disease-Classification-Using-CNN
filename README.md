# 📌 Live Multi-Class Skin Disease Classification Using CNN

## 📖 Overview

This project is a deep learning–based web application that classifies multiple skin diseases using a Convolutional Neural Network (CNN). Users can upload skin images and receive predictions through a Flask-based interface. The system demonstrates a complete end-to-end AI pipeline from model training to deployment. This project showcases medical image classification using CNN, Flask web application integration, real-time inference (local webcam support), and model deployment readiness.

## 🎯 Objectives

Build a CNN model to classify multiple skin diseases, provide an easy-to-use web interface for prediction, enable real-time inference using webcam (local environment), and demonstrate a complete deep learning deployment pipeline.

## 🧠 Model Details

Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input: Skin disease images
Output: Predicted disease class
Saved Model: best_skin_model.h5
The CNN learns visual features such as texture patterns, color variations, lesion shapes, and skin abnormalities.

## 🗂️ Project Structure

Live-Multi-class-Skin-Disease-Classification-Using-CNN/
│
├── dataset/                      # Training dataset
├── model/
│   └── best_skin_model.h5        # Trained CNN model
├── static/
│   ├── home.css
│   ├── result.css
│   └── uploads/                  # Uploaded images
├── templates/
│   ├── home.html
│   └── result.html
├── app.py                        # Flask backend
├── skin.ipynb                    # Model training notebook
├── opencv.py                     # Webcam logic (local)
├── requirements.txt

## ⚙️ Features

1️⃣ Image Upload Prediction
Upload a skin image, the model predicts the disease class, and displays prediction with confidence score.
2️⃣ Real-Time Prediction (Local Only)
Uses OpenCV to capture webcam frames, performs frame-by-frame classification, and displays live predictions on screen. Note: Live webcam feature works only on local machines. Cloud platforms do not support camera hardware.
3️⃣ Patient Information Collection
User inputs include name, age, phone number, and optional email.

## 🛠️ Technologies Used

Python, TensorFlow / Keras, Flask, OpenCV, NumPy, HTML, CSS.

## 📊 Workflow

Dataset preprocessing, image resizing and normalization, CNN model training, model saving (.h5), Flask backend integration, web interface for predictions, and deployment.

## 📥 Installation

Step 1 — Clone Repository
git clone https://github.com/Srikhar07/Live-Multi-class-Skin-Disease-Classification-Using-CNN.git
cd Live-Multi-class-Skin-Disease-Classification-Using-CNN
Step 2 — Install Dependencies
pip install -r requirements.txt
Step 3 — Run Application
python app.py
Open in browser: http://127.0.0.1:5000/

## 📦 Requirements

flask
tensorflow
numpy
opencv-python
h5py

## 🚀 Deployment Notes

This project can be deployed on Render, Railway, or Heroku. Image upload prediction works in cloud environments. Webcam prediction works only locally because servers do not have camera access.

## 📈 Model Performance

The CNN model performs multi-class classification based on image features. Performance depends on image quality, lighting conditions, dataset size, and dataset balance.

## 🔬 Real-World Applications

Early screening support system, dermatology assistance tool, telemedicine support, and AI-based medical demonstration application.

## ⚠️ Limitations

Not a replacement for professional medical diagnosis. Accuracy depends on dataset quality. Webcam prediction is not supported in cloud deployment. Performance may vary on low-resource servers.
