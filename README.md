# 🌍 Hybrid Air Quality Prediction Model  

A Machine Learning project to predict **Air Quality Index (AQI)** using pollutant data (PM2.5, PM10, NO2, CO, SO2, O3).  
Deployed as a **Flask web app** with real-time AQI predictions and category labels (*Good, Satisfactory, Moderate*).  

---

## 🚀 Features
- Data preprocessing (handling missing values, PCA, label encoding).  
- ML models: Linear Regression, Random Forest, Hybrid RF + Passive Aggressive Classifier.  
- Flask web app for user input and real-time AQI prediction.  
- Performance evaluation using Accuracy, Precision, Recall, F1-score.  

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask  
- **ML Libraries:** Pandas, Scikit-learn, Numpy  
- **Frontend:** HTML, CSS  
- **Dataset:** CSV pollutant data with AQI categories  

---

## 📂 Project Structure
CODE/
│── app.py # Flask application
│── MainFile.py # ML pipeline & model training
│── Dataset.csv # Dataset used
│── templates/ # HTML templates for web UI
│── random_forest_model.pkl # Trained Random Forest model
│── linear_regression_model.pkl
│── data.pkl
│── UserData.py

