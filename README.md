# ❤️ Cardiovascular Disease Prediction

This project applies **Machine Learning techniques** to predict the likelihood of **cardiovascular disease (CVD)** using patient health data.  
The goal is to support early detection and intervention, helping healthcare providers make data-driven decisions to reduce risk.  

---

## 📌 Project Overview
- **Domain:** Healthcare Analytics / Predictive Modeling  
- **Techniques:** Supervised Learning (Classification)  
- **Objective:** Build and compare machine learning models to predict cardiovascular disease risk based on clinical attributes.  

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - Scikit-learn (model training & evaluation)  
  - Pandas & NumPy (data manipulation)  
  - Matplotlib & Seaborn (visualization)  
- **Environment:** Jupyter Notebook  

---


---

## 📊 Dataset
- **Source:**  https://archive.ics.uci.edu/dataset/45/heart+disease 
- **Size:** 70,000+ patient records  
- **Features include:**  
  - Age, Gender, Height, Weight  
  - Blood Pressure (systolic/diastolic)  
  - Cholesterol, Glucose  
  - Smoking, Alcohol intake, Physical activity  
- **Target Variable:** `cardio` (1 = disease present, 0 = no disease)  

---

## 🧹 Data Preprocessing
- Removed duplicates and missing values.  
- Normalized continuous variables (e.g., Age, BMI).  
- Encoded categorical variables.  
- Split dataset into **Train (70%)** and **Test (30%)**.  

---

## 🤖 Models Implemented
- Logistic Regression  
- k-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

---

## 📈 Results
- **Best Model:** Random Forest  
- **Accuracy:** ~84%  
- **Recall (Class 0):** 91% (healthy individuals)  
- **Recall (Class 1):** 75% (patients with CVD)  
- **Precision (Class 1):** 75%  


---

## 🚀 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sneh6677/Cardio_vascular_disease_prediction.git
   cd Cardio_vascular_disease_prediction

