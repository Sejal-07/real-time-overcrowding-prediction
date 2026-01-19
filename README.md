<div align="center">

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![API](https://img.shields.io/badge/API-FastAPI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CV](https://img.shields.io/badge/CV-OpenCV-orange)
![ML](https://img.shields.io/badge/ML-Scikit--learn-purple)

</div>


# 🚦 Real-Time Overcrowding Prediction 

An **AI-powered smart transportation solution** designed to **predict overcrowding in real time**, optimize passenger flow, and assist authorities in making **data-driven decisions**.  
This system enhances commuter experience by reducing congestion, wait times, and uncertainty in public transport systems.

---

## 📌 Project Overview

Public transportation systems often suffer from unpredictable overcrowding, leading to delays, discomfort, and safety concerns.  
This project leverages **Machine Learning and Time-Series Analysis** to forecast crowd levels and proactively manage congestion.

### Key Objectives
- Predict real-time crowd density at transport stops and vehicles
- Alert passengers and authorities before overcrowding occurs
- Suggest alternative routes or time slots
- Improve operational efficiency and commuter safety

---

## ✨ Features

- 🔮 **Real-Time Overcrowding Prediction**
- 📊 **Crowd Level Classification** (Low / Medium / High)
- 🚨 **Automated Alerts & Notifications**
- 🧭 **Smart Route & Timing Suggestions**
- 🎟️ **QR-based Ticket Data Integration**
- 📈 **Admin Dashboard for Analytics**

---

## 🧠 System Architecture
<img width="1536" height="1024" alt="ChatGPT Image Jan 12, 2026, 06_29_40 PM" src="https://github.com/user-attachments/assets/52a86b65-c3ad-4068-a1ee-f81ee6d9fe3c" />


---
## 🧠 Machine Learning Workflow

1. Data Collection (QR ticket scans, timestamps, route & stop IDs)
2. Data Preprocessing & Feature Engineering
3. Model Training (Classification / Time-Series models)
4. Real-Time Prediction
5. Crowd Level Output (Low / Medium / High)
6. Visualization on Dashboard

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI
- REST APIs
### Machine Learning
- Scikit-learn
- Model: Random Forest
- NumPy, Pandas
### Computer Vision
- OpenCV (webcam access & frame processing)
-	QR Code detection using OpenCV QRCodeDetector
### Dashboard
- Streamlit

---

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/real-time-overcrowding-prediction.git
cd real-time-overcrowding-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn backend.api:app --reload

```
Access API documentation at:
```bash
http://127.0.0.1:8000/docs
```
---
## 📸 Output Preview

The screenshot displays real-time passenger count tracking and ML-based overcrowding prediction, visualized through an interactive dashboard for quick decision-making.

<img width="1783" height="828" alt="Screenshot 2026-01-19 145305" src="https://github.com/user-attachments/assets/5483f088-0ec1-4714-9eb3-4a09322cc9c0" />















