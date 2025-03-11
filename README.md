# 💳 Credit Risk Assessment using Probability & Statistics
> A Machine Learning-based web app to assess whether a loan applicant is at **High Risk** or **Low Risk**, using **statistical analysis** and **classification models** trained on historical loan data.

---

## 📌 Live Demo

🎯 Try the app now 👉 [Credit Risk Assessment Web App](https://creditriskassessment-kbvenp72yhhvivunbxwsmh.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditriskassessment-kbvenp72yhhvivunbxwsmh.streamlit.app/)

--------

## 🧠 Project Overview

This project predicts the **creditworthiness of a customer** using probability & classification techniques in Machine Learning. It helps banks and financial institutions to determine the **risk level** of loan applicants based on historical data.

The application takes various inputs like:
- Age
- Income
- Loan Amount
- Credit Score
- Marital Status
- Employment Type  
...and more!

It then performs real-time predictions and visual feedback using trained ML models.

---------
## 📊 Algorithms Used

We trained and evaluated the following models:

| Algorithm                | Accuracy |
|--------------------------|----------|
| Logistic Regression      | ✅ 84.5% |
| Random Forest Classifier | ✅ **89.7%** (Best) |
| Support Vector Machine   | ✅ 86.3% |
| K-Nearest Neighbors      | ✅ 81.2% |

📌 **Random Forest Classifier** was selected as the final model for deployment due to its superior performance and handling of mixed-type features.

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit (Python Web Framework)
- **Backend:** Machine Learning (Scikit-learn, Pandas, Numpy)
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud
- **Model Persistence:** Joblib

---

## 💻 How to Run Locally

Follow these steps to run the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-risk-assessment.git
cd credit-risk-assessment
