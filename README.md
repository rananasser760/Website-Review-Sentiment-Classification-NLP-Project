# NLP Review Sentiment Classification

*Kaggle Neural Networks Competition – Faculty of Computer and Information Sciences, Ain Shams University*

---

## 📌 Project Overview

This project is an **NLP Review Sentiment Classification system**.

The goal of this project is to help websites understand their overall ranking based on users’ reviews.  
By developing deep learning models, we automatically convert a user’s written comment into one of **five sentiment classes**:

**(Excellent, Very Good, Good, Bad, Very Bad)**

The system is designed to accurately classify each review while handling class imbalance and real-world noisy text.

---

## 🧠 Why This Project?

User reviews are subjective, noisy, and often imbalanced.  
This project focuses on:
- Understanding **contextual meaning**
- Handling **ordinal sentiment levels**
- Improving performance on **minority classes**
- Building **robust ensemble models**

---

## 🔹 Text Preprocessing & Feature Engineering

To improve model generalization and reduce noise, the following preprocessing steps were applied:

- Lowercasing text
- Removing URLs and HTML tags
- Removing special characters and extra spaces
- Expanding contractions (e.g., *can't → cannot*)
- Text normalization

📌 **Why?**  
Clean and normalized text helps models focus on semantic meaning instead of irrelevant noise.

---

## 🔹 Data Augmentation & Class Balancing

The dataset was imbalanced, so minority classes were augmented using:
- Synonym replacement
- Word deletion
- Word swapping
- Word insertion

Additionally:
- Class weights
- Balanced Accuracy metric
- Weighted sampling

📌 **Why?**  
To prevent the model from being biased toward majority classes and to improve performance across all sentiment levels.

---

## 🔹 Models Used

### 🔹 1. Transformer-Based Model (DeBERTa – PyTorch)

- Fine-tuned **DeBERTa-v3 Large**
- Disentangled attention mechanism
- Strong contextual understanding
- Layer-wise learning rate decay (LLRD)
- Custom **Ordinal + Focal Loss**

📌 **Why DeBERTa?**  
It performs exceptionally well on sentiment and contextual NLP tasks.

📌 **Why Ordinal + Focal Loss?**
- Focal Loss focuses on hard samples
- Ordinal Loss respects sentiment order (Very Bad → Excellent)

---

### 🔹 2. Deep Learning Models from Scratch (TensorFlow)

Multiple architectures were trained to capture different linguistic patterns:

- **Transformer Encoder**  
  → Captures global context

- **BiLSTM**  
  → Learns sequential and bidirectional dependencies

- **CNN + BiLSTM**  
  → Extracts local n-gram features + sequence modeling

- **BiGRU with Attention**  
  → Focuses on the most important words in each review

📌 **Why multiple models?**  
Each architecture captures different aspects of language.

---

## 🔹 Ensemble Learning

Final predictions were generated using a **weighted ensemble**, where:
- Each model was weighted based on its validation **Balanced Accuracy**
- Probabilities were combined to produce final predictions

📌 **Why ensemble?**  
Ensembles improve stability, robustness, and overall performance.

---

## 📊 Evaluation Metric

- **Balanced Accuracy**
  
Chosen due to class imbalance, ensuring fair evaluation across all classes.

---

## 🛠 Tech Stack

- **Python**
- **PyTorch**
- **TensorFlow / Keras**
- **HuggingFace Transformers**
- **Scikit-learn**
- **NumPy & Pandas**

---

## 🚀 Key Learnings

This project strengthened my experience in:
- NLP pipelines
- Transformer fine-tuning
- Sequence models (LSTM, GRU)
- Attention mechanisms
- Ensemble learning
- Handling real-world imbalanced datasets
- Working with both PyTorch and TensorFlow

---

## 📌 Competition Achievement

🥇 **1st Place – Public Leaderboard**  
🥉 **3rd Place – Private Leaderboard**

---
