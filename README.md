
# 🐶🐱 End-to-End Enterprise MLOps Pipeline  
## Binary Image Classification – Cats vs Dogs

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-blue)
![DVC](https://img.shields.io/badge/DVC-DataVersioning-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHubActions-black)

---

## 📌 Executive Summary

This project implements a **production-grade, end-to-end MLOps pipeline** for a binary image classification use case (Cats vs Dogs), designed for a pet adoption platform.

The pipeline demonstrates:

- Data & code versioning
- Model training & experiment tracking
- Model packaging & containerization
- Continuous Integration (CI)
- Continuous Deployment (CD)
- Monitoring & logging

The system is designed to be **reproducible, scalable, and deployment-ready** following industry MLOps best practices.

---

## 🏗️ High-Level Architecture

```
Kaggle Dataset
      ↓
DVC (Data Versioning)
      ↓
Model Training (PyTorch)
      ↓
MLflow (Experiment Tracking)
      ↓
Model Artifact (.pt)
      ↓
FastAPI Inference API
      ↓
Docker Image
      ↓
GitHub Actions (CI/CD)
      ↓
Docker Hub
      ↓
Docker Compose Deployment
```

---

## 🧰 Technology Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.10 |
| Deep Learning | PyTorch |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| API Layer | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Docker Compose |
| Testing | Pytest |
| Monitoring | Logging + Request Counters |

---

## 📂 Repository Structure

```
cats-dogs-mlops/
│
├── data/
├── src/
│   ├── data/
│   ├── model/
│   └── utils/
│
├── app/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

---

# 🔹 Model Development (M1)

## Data Processing
- Images resized to 224x224 RGB
- Dataset split: 80% Train / 10% Validation / 10% Test
- Versioned using DVC

## Model Architecture
Baseline CNN implemented in PyTorch:

- Convolution Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Sigmoid Output

Model artifact saved as:

```
model.pt
```

## Experiment Tracking
MLflow logs:
- Hyperparameters
- Accuracy metrics
- Model artifacts

Run MLflow UI:

```
mlflow ui
```

---

# 🔹 Packaging & Containerization (M2)

## API Endpoints

### Health Check
```
GET /health
```

### Prediction
```
POST /predict
```

Response Example:

```json
{
  "probability": 0.91,
  "label": "dog"
}
```

## Docker

Build:
```
docker build -t catsdogs:latest .
```

Run:
```
docker run -p 8000:8000 catsdogs
```

---

# 🔹 Continuous Integration (M3)

GitHub Actions pipeline performs:

1. Code checkout
2. Dependency installation
3. Unit testing (pytest)
4. Docker build
5. Docker image push to Docker Hub

Pipeline file:

```
.github/workflows/ci.yml
```

---

# 🔹 Continuous Deployment (M4)

Deployment via Docker Compose:

```
docker-compose up -d
```

Includes smoke test validation:

```
curl http://localhost:8000/health
```

---

# 🔹 Monitoring & Observability (M5)

Features:

- Request logging
- Prediction logging
- Request counters
- Error tracking readiness

Example Log:

```
INFO: Prediction request #25
```

---

# 🧪 Local Setup Guide

## Clone Repository
```
git clone https://github.com/<your-username>/cats-dogs-mlops.git
cd cats-dogs-mlops
```

## Create Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Train Model
```
python src/model/train.py
```

## Run API
```
uvicorn app.main:app --reload
```

---

# 🧪 Testing

Run tests:

```
pytest
```

---

# 📊 Production Readiness Highlights

- Fully reproducible training pipeline
- Containerized inference service
- Automated CI/CD
- Versioned dataset
- Tracked experiments
- Structured logging

---

# 🚀 Future Enhancements

- Kubernetes deployment
- Prometheus & Grafana integration
- Transfer learning (ResNet/EfficientNet)
- Model drift detection
- Automated retraining pipeline 

---

# ⭐ Conclusion

This repository demonstrates a complete enterprise-grade MLOps lifecycle from data ingestion to automated deployment and monitoring, aligned with industry best practices.

