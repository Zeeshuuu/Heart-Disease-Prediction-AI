# Heart-Disease-Prediction-AI
A machine learning–based diagnostic system that predicts the risk of heart disease using medical data. Built with Python, Scikit-learn, Streamlit, and SHAP for model interpretability.


# About Me
Hi, I'm **Zeeshan Memon** — a **Computer Science Graduate** and **AI & ML Enthusiast**.  
I love building intelligent systems that use data and machine learning to solve real-world problems.  
This project demonstrates my skills in **machine learning**, **data preprocessing**, and **explainable AI**, deployed through a **Streamlit dashboard**.

#  Project Overview

The **Heart Disease Prediction AI** uses the **UCI Heart Disease Dataset** to predict the risk of heart disease in patients.  
It applies multiple machine learning models and uses **SHAP (SHapley Additive Explanations)** for transparent and interpretable predictions.

 Built and trained models using **Scikit-learn**  
 Deployed an interactive dashboard with **Streamlit**  
 Explained model predictions using **SHAP visualizations**  
 Achieved **85.33% accuracy** and **0.91 AUC** with **Random Forest Classifier**

#  Project Highlights

- Preprocessed medical data and handled missing values  
- Performed one-hot encoding for categorical features  
- Scaled data using **StandardScaler**  
- Trained and compared 3 models:
  - Logistic Regression  
  - Gradient Boosting  
  - **Random Forest (Best Model)**
- Visualized model performance using confusion matrices and AUC  
- Implemented SHAP explainability for transparent AI decisions  
- Integrated results in an interactive **Streamlit dashboard**

#  Tech Stack

| Category | Tools |
|-----------|--------|
| **Language** | Python  |
| **ML Libraries** | scikit-learn, SHAP, joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Data Handling** | Pandas, NumPy |

#  Model Performance

| Model | Accuracy | AUC Score |
|--------|-----------|-----------|
| Logistic Regression | 83.19% | 0.90 |
| Gradient Boosting | 84.26% | 0.91 |
| **Random Forest (Best)** | **85.33%** | **0.91** |

 **Best Model:** Random Forest Classifier  
  Model & Scaler saved as `.pkl` files for deployment.

#  File Structure

Heart_Disease_AI/
│
├── heart_dashboard.py # Streamlit dashboard app
├── heart_disease_model.pkl # Model training & evaluation script
├── heart_disease_uci.csv # Dataset
├── scaler.pkl
├── requirements.txt
└── README.md


If you like this project, give it a star!
It helps others find and learn from it.
