

# Hospital Readmission Risk Prediction with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20%26%20LIME-green.svg)

## 📌 Project Overview
This project aims to predict the risk of hospital readmission for diabetes patients within 30 days of discharge. Using the **Diabetes 130-US hospitals dataset**, we developed a high-precision machine learning pipeline that not only predicts risk but also provides clinical interpretability using SHAP and LIME.

The goal is to provide healthcare professionals with actionable insights to reduce readmission rates, optimize resource allocation, and improve patient outcomes.

---

## 📊 Key Results
- **Model:** XGBoost Classifier
- **Primary Metric (AUC-ROC):** 0.78 (Current baseline after data cleaning)
- **Interpretability:** Integrated SHAP summary plots for global feature importance and LIME for local, patient-specific explanations.

---

## 🛠 Tech Stack
* **Modeling:** XGBoost, Scikit-learn
* **Optimization:** Optuna (Hyperparameter Tuning)
* **Explainable AI:** SHAP, LIME
* **Data Handling:** Pandas, NumPy
* **Reporting:** Groq API (Llama 3.3 70B) for automated clinical report generation

---

## 🔍 Explainability & Clinical Insights

### Global Interpretability (SHAP)
We use SHAP to understand which features drive the model's overall logic. 
* **Prior Hospital Admissions:** The strongest predictor of future readmission.
* **Number of Medications:** High polypharmacy correlates with increased risk.
* **Discharge Disposition:** Patients discharged to specialized facilities show different risk profiles compared to home discharges.



### Local Interpretability (LIME)
For every high-risk prediction, the system generates a LIME explanation to show *why* that specific patient is at risk, allowing doctors to see the weight of features like "Number of Diagnoses" or "Emergency Visits" for an individual case.

---

## 🧼 Data Preprocessing Highlights
To ensure the model is clinically valid, the following steps were taken:
1.  **Mortality Filtering:** Removed patients with discharge IDs corresponding to "Expired" or "Hospice Care" (IDs 11, 13, 14, 19, 20, 21) as they are not eligible for readmission.
2.  **Feature Engineering:** Created `total_visits` and `severity_index` to capture complex patient histories.
3.  **Handling Missing Values:** Managed '?' and null values in medical records (race, weight, payer_code).
4.  **Categorical Encoding:** Applied One-Hot Encoding for medical diagnosis codes (ICD-9).

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/hospital-readmission-xai.git](https://github.com/yourusername/hospital-readmission-xai.git)

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the analysis notebook or script:
    ```bash
    jupyter notebook data.ipynb
    ```

-----

## 📝 Automated Reporting

The project includes a specialized module that takes model outputs and generates a **Medical Analysis & Readmission Report** in PDF format using the Groq API. This report translates technical SHAP/LIME values into clinical recommendations for hospital staff.

-----

## ⚖️ Fairness & Ethics

  * **Bias Mitigation:** Regularization (L1/L2) is used to prevent the model from over-relying on sensitive features.
  * **Audit-Ready:** The interpretability layer allows for auditing model decisions to ensure no discriminatory patterns based on demographic data.
