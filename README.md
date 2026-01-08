# 🛒 SuperKart Sales Forecasting – Advanced ML & MLOps

This repository contains an **end-to-end machine learning and MLOps pipeline** for forecasting product-level sales revenue across SuperKart retail stores. The project demonstrates how a machine learning model can be taken from data preparation to production deployment using modern MLOps practices.

---

## 📌 Business Problem
Accurate sales forecasting is critical for effective inventory planning, supply chain optimization, and regional sales strategy. SuperKart requires a scalable and automated solution to predict sales revenue based on product attributes and store characteristics.

---

## 🎯 Project Objective
To design and implement a **fully automated MLOps pipeline** that:
- Trains a machine learning model for sales revenue prediction  
- Tracks experiments and metrics using MLflow  
- Versions datasets and models using Hugging Face Hub  
- Deploys a production-ready Streamlit application using Docker  
- Automates the workflow using GitHub Actions CI/CD  

---

## 🧠 Machine Learning Approach
- **Problem Type:** Regression  
- **Model Used:** XGBoost Regressor  
- **Target Variable:** `Product_Store_Sales_Total`  
- **Evaluation Metrics:** RMSE, MAE, R²  

---

## 🏗️ Project Structure

```text
superkart_project/
│
├── model_building/
│   ├── data_register.py
│   ├── prep.py
│   ├── train.py
│   └── model_prod.joblib
│
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── hosting/
│   └── hosting.py
│
├── requirements.txt
└── README.md
