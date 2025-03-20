# Deep-Learning-on-Intrusion-Detection-System
This project implements a multi-model Intrusion Detection System (IDS) using various deep learning and machine learning techniques for detecting and classifying network intrusions. The goal is to compare different AI-driven approaches for network security.

# Intrusion Detection System (IDS) Using AI Models

## Overview
This project implements multiple AI-driven approaches for **Intrusion Detection Systems (IDS)** to classify and detect malicious network activity. The objective is to compare different deep learning and machine learning models for network security and anomaly detection.

## Implemented Models

### 1. **TabNet Classifier**
- A deep learning model optimized for tabular data, trained to classify network intrusions.

### 2. **Deep Q-Network (DQN) - Reinforcement Learning Based IDS**
- Uses an RL agent to learn optimal security actions based on network traffic data.

### 3. **Autoencoder for Anomaly Detection**
- Learns to reconstruct normal network traffic patterns and detects anomalies based on high reconstruction errors.

### 4. **Generative Adversarial Network (GAN) for Anomaly Detection**
- A GAN-based approach where the generator learns to mimic normal traffic and the discriminator identifies anomalies.

### 5. **Variational Autoencoder (VAE) + Wasserstein GAN (WGAN)**
- A combination of VAE and WGAN to improve feature learning and generate better synthetic representations for anomaly detection.

### 6. **Hybrid CNN + RNN + LSTM Model**
- Uses **CNN** for feature extraction, **RNN** for sequential dependencies, and **LSTM** for capturing long-term patterns in network traffic.

### 7. **Attention-based CNN-Transformer-LSTM Model**
- A hybrid model that integrates **convolutional layers, transformers (self-attention), and LSTM** to capture both spatial and temporal dependencies in network traffic.

## Dataset Used
- **CICIDS2017** dataset is used for training and evaluation.

## Installation & Requirements

### **Dependencies:**
Ensure you have the following libraries installed:
```bash
pip install numpy pandas torch torchvision tabnet gym sklearn
```

### **Running the Models:**
Execute the respective scripts to train and evaluate each model:
```bash
python train_tabnet.py
python train_dqn.py
python train_autoencoder.py
python train_gan.py
python train_cnn_rnn_lstm.py
python train_transformer_lstm.py
```

## Results & Evaluation
- Model performance is evaluated using metrics such as **Accuracy, Precision, Recall, F1-score, and AUC-ROC.**
- The hybrid models (CNN + RNN + LSTM and CNN-Transformer-LSTM) showed improved accuracy due to better feature extraction and sequential learning.

## Future Work
- Implement **graph neural networks (GNNs)** for better anomaly detection.
- Improve **GAN training stability** with WGAN-GP.
- Enhance dataset preprocessing with **feature selection and dimensionality reduction techniques**.

## Contributors
- **Hemang Sharma** ([GitHub](https://github.com/Hemu-7))

## License
This project is licensed under the MIT License.
