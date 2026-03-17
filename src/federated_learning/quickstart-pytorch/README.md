# 🚀 Federated Learning on CIFAR-10 using Flower & PyTorch

## 📌 Project Overview
This project implements a federated learning system using Flower and PyTorch on the CIFAR-10 dataset.

Multiple clients collaboratively train a global model without sharing raw data.

---

## 🧠 Model Improvements
- Added Batch Normalization for stability
- Tuned learning rate and epochs
- Reduced overfitting using Dropout

---

## 📊 Results
- Final Accuracy: **~76%**
- Training Rounds: 20
- Dataset: CIFAR-10

---

## 📈 Key Observations
- Accuracy improved steadily across rounds
- Training stabilized after BatchNorm
- Overfitting controlled with Dropout

---

---
```markdown
## ⚙️ How to Run

```bash
pip install -e .
flwr run

## 📁 Project Structure

🔗 Reference

Flower Documentation: https://flower.ai/docs/
src/federated_learning/quickstart-pytorch/
