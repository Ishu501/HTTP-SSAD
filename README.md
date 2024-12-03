# HTTP-SSAD
# HTTP Anomaly Detection using BERT-Autoencoders
  This project uses a BERT-based model and autoencoders for detecting anomalies in HTTP traffic. The system classifies HTTP    requests as benign or malicious using advanced NLP techniques and unsupervised learning.

# Features
BERT Embeddings for feature extraction from HTTP requests.
Autoencoder-based Anomaly Detection for identifying malicious traffic.
Real-Time Detection via a FastAPI server endpoint.

# Datasets:
  CSIC-2010
  UNSW-NB15
  Malicious URL
  ISCX-URL-2016

# Installation:
  Clone the repository: git clone <repo_url> ____  cd <repo_folder>
  Install dependencies: pip install -r requirements.txt
  Train the model: python anomaly_detection.py
  Start FastAPI server for real-time detection: uvicorn anomaly_detection:app --reload

# Usage:
  Use the FastAPI endpoint to detect anomalies in HTTP requests. Example request:
curl -X POST "http://127.0.0.1:8000/detect" -H "Content-Type: application/json" -d '{"requests": ["GET /testHTTP/1.1", "POST /malicious/path HTTP/1.1"]}'

# Evaluation Metrics:
  F1 Score: Measures accuracy of classification.
  False Positive Rate (FPR) at 90% and 99% thresholds.
# Results:
  The model is evaluated on four datasets, showing that BERT consistently outperforms TF-IDF and BoW methods.
