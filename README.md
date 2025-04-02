

# 🩺 Lung Cancer Prediction System

## 📌 Project Overview
This is a **machine learning-based web application** developed using **Streamlit** and **Logistic Regression** to predict the likelihood of lung cancer based on user inputs. The model considers various factors such as smoking habits, medical history, and respiratory symptoms to assess potential risk.

## 🔥 Features
- **User-friendly UI** built with Streamlit
- **Real-time lung cancer risk prediction** based on user inputs
- **Uses Logistic Regression** for classification
- **Data preprocessing & handling missing values**
- **Stylized output** with icons, colors, and alerts for better readability

## 🏗️ Tech Stack
- **Python**
- **Streamlit (Frontend)**
- **Pandas (Data Handling)**
- **Scikit-learn (ML Model)**

## 📂 Dataset
The dataset contains medical and lifestyle-related features:
| Feature               | Type    | Description |
|----------------------|--------|-------------|
| AGE                  | Integer | Age of the patient |
| SMOKING              | Boolean | Smoking habit (Yes/No) |
| YELLOW_FINGERS       | Boolean | Yellow fingers due to smoking (Yes/No) |
| ANXIETY              | Boolean | Presence of anxiety (Yes/No) |
| PEER_PRESSURE        | Boolean | Influence of peer pressure (Yes/No) |
| CHRONIC DISEASE      | Boolean | Presence of chronic disease (Yes/No) |
| FATIGUE              | Boolean | Frequent fatigue (Yes/No) |
| ALLERGY              | Boolean | Any allergies (Yes/No) |
| WHEEZING             | Boolean | Wheezing symptoms (Yes/No) |
| ALCOHOL CONSUMING    | Boolean | Alcohol consumption habit (Yes/No) |
| COUGHING             | Boolean | Frequent coughing (Yes/No) |
| SHORTNESS OF BREATH  | Boolean | Shortness of breath (Yes/No) |
| SWALLOWING DIFFICULTY| Boolean | Difficulty in swallowing (Yes/No) |
| CHEST PAIN           | Boolean | Chest pain symptoms (Yes/No) |

The **target variable** is `LUNG_CANCER` (1 = High Risk, 0 = Low Risk).

## 📥 Installation & Setup

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```sh
streamlit run app.py
```

## 🛠️ Model Training & Deployment
- The dataset is preprocessed to handle missing values (mode/median imputation)
- **Label Encoding** is applied to categorical variables
- **Logistic Regression** is trained on 80% of the data and tested on 20%
- The model is integrated into the Streamlit app for real-time prediction
- The model was found to be 93.54838709677419 % accurate

## 📊 Output & Results
- If the model predicts **low risk**, the app shows a **green success message** ✅
- If the model predicts **high risk**, the app displays a **red warning message** ⚠️

## 🎨 UI Enhancements
- **Progress indicators** to simulate model processing
- **Colored alerts** for risk levels
- **Icons & images** to improve user engagement



## ⚠️ Disclaimer
This tool is for **informational purposes only** and **does not provide medical diagnosis**. Consult a professional for medical advice.



