📦 Teachable Machine – Streamlit Version

A simple, clean, and fully interactive image classification web-app built using Streamlit + TensorFlow.
This project works just like Google’s Teachable Machine, allowing you to:

Create custom classes

Upload images

Train a deep learning model

Predict new images instantly

All inside your browser — with no coding required.

🚀 Features
1️⃣ Add Unlimited Classes

Create your own labels (e.g., Cat, Dog, Apple, Car).
Each class automatically gets its own dataset folder.

2️⃣ Upload Images for Every Class

Upload multiple images per class.
Images are stored cleanly inside:

dataset/class_name/

3️⃣ Train a CNN Model Inside the App

Model uses a Convolutional Neural Network (CNN) with:

3× Conv2D layers

MaxPooling

Dropout

Dense classification head

Plus real-time logs + accuracy/loss charts using Streamlit.

4️⃣ Live Prediction

Upload any image to test the trained model.
App shows:

Predicted class

Confidence %

Probability bar chart

5️⃣ Session-State Powered

App remembers:

Classes

Uploaded images

Model

Training history

Even when you switch tabs.

6️⃣ Reset App

Clear everything and start fresh with one click.

🛠️ Tech Stack

Python 3.9+

Streamlit

TensorFlow / Keras

NumPy

Pandas

Pillow (PIL)

📁 Project Structure
├── dataset/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── trained_model.keras
├── app.py
└── README.md

🎯 Why This Project?

This project is perfect for:

Beginners learning Machine Learning

Students making ML-based projects

Developers creating custom classifiers

Anyone wanting a Teachable Machine alternative in Python

▶️ How to Run
pip install -r requirements.txt
streamlit run app.py

📌 Notes

Model trains on CPU/GPU depending on your system

Uses real-time callbacks to generate charts

Dataset builds automatically
