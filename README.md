# AI-REVEAL
A deep learning–based project that detects whether an image or video is AI-generated or real using Convolutional Neural Networks (CNNs) and FFT-based frequency analysis. The system identifies subtle artifacts such as distorted hands, unnatural textures, and inconsistent lighting patterns that commonly appear in AI-generated content.


🚀 Features

Detects AI-generated vs. real images and videos

Uses a hybrid CNN + FFT model for higher accuracy

Trains on datasets of real and synthetic hands, faces, and objects

Includes a beautiful PyQt5 desktop UI for predictions

Visualizes prediction confidence with graphs and probability scores

Supports model training, testing, and evaluation


🧩 Tech Stack

Python (TensorFlow / Keras / NumPy / OpenCV)

PyQt5 – for aesthetic GUI design

Matplotlib – for accuracy/loss visualization

FFT (Fast Fourier Transform) – for frequency-based artifact detection


🧰 Dataset

Real and AI-generated images (collected from Kaggle & Hugging Face)

Includes various categories like hands, faces, and natural scenes

Preprocessed and split into train and validation sets


📊 Model Workflow

Image Preprocessing – Resize, normalize, and optionally apply FFT

Model Training – CNN learns spatial + frequency features

Validation – Evaluate on unseen data

Prediction – Classify input as “AI-Generated” or “Real”

Visualization – Show confidence graph and label in GUI


🎨 User Interface

A modern PyQt5 interface lets users:

Upload image/video files

View real-time prediction with probability

See visualization of model confidence


🏁 Future Scope

Extend detection to AI-generated audio and text

Integrate with a web-based dashboard

Use Vision Transformers (ViTs) for improved detection

the project is still in progress 
