# ❤️ CardioCare — Cardiovascular Disease Prediction Using Machine Learning

CardioCare is a machine learning mini-project that predicts cardiovascular disease outcomes using patient-related health features. It compares five machine learning classification algorithms and provides an interactive web dashboard built using Streamlit.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, comparison, and deployment.

🔗 **Live Demo:** https://cardiocare-cvd.streamlit.app/

## 🎯 Objectives

- Analyze and understand cardiovascular disease data.
- Perform data preprocessing and feature selection.
- Implement and compare five machine learning algorithms.
- Evaluate model performance using classification metrics.
- Deploy an interactive prediction dashboard using Streamlit.

## 📊 Dataset

- **Dataset:** Cardiovascular Disease Dataset
- **Total Records:** 1,000
- **Input Features:** 12
- **Target Variable:** `target`
- **Problem Type:** Binary Classification

The dataset includes demographic, clinical, and other health-related features, such as age, gender, chest pain, resting blood pressure, serum cholesterol, maximum heart rate, and exercise-induced angina.

The `patientid` column is excluded from model training because it is an identifier rather than a predictive feature.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Machine learning and model evaluation
- **Matplotlib & Seaborn** — Data visualization
- **Streamlit** — Interactive web dashboard
- **Joblib** — Saving and loading trained models

## 🤖 Machine Learning Algorithms

Five classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

## 📈 Model Performance

The models were evaluated using accuracy, precision, recall, and F1-score.

| Algorithm | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 98.00% | 98.29% |
| Decision Tree | 97.50% | 97.82% |
| Random Forest | 98.50% | 98.71% |
| K-Nearest Neighbors | 91.50% | 92.70% |
| Support Vector Machine | 97.00% | 97.46% |

**Best-performing model: Random Forest**

Random Forest was selected because it achieved the highest accuracy (98.50%) and F1-score (98.71%) among the five evaluated models.

These results apply to the project's evaluation dataset and do not establish clinical reliability.

## 🖥️ Dashboard Features

- Input fields for patient-related health attributes
- Machine learning-based prediction
- Display of predicted class probabilities
- Model performance information
- Simple, interactive user interface

## ⚙️ Project Workflow

1. Dataset selection and understanding
2. Data inspection and preprocessing
3. Feature selection and scaling
4. Exploratory data analysis
5. Training five classification models
6. Evaluating and comparing model performance
7. Selecting the best-performing model
8. Saving the trained model and scaler
9. Integrating the model with Streamlit
10. Deploying the interactive dashboard

## 🚀 Run the Project Locally

**1. Clone the repository**

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

**2. Navigate to the project folder**

```bash
cd CardioCare
```

**3. Install the required libraries**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

Replace `app.py` with your actual Streamlit application filename if it is different.

Ensure that the dataset and saved model files required by the application are available in the appropriate locations.

## 🔮 Future Scope

- Validate the model using larger and independent datasets.
- Explore explainable AI techniques such as SHAP.
- Perform cross-validation and additional model evaluation.
- Improve dashboard design and input validation.
- Investigate the model's generalization across different populations.

## ⚠️ Disclaimer

CardioCare is an academic and educational project. It is not a medically validated diagnostic tool and should not be used to diagnose cardiovascular disease or make healthcare decisions. Always consult a qualified healthcare professional for medical advice.

## 👩‍💻 Project Team

**B.Tech in Data Science | MGM University**  
Chhatrapati Sambhajinagar, Maharashtra  
Academic Year: 2026–2027

- Aliza Syeda
- Sakshi Chilka
- Prerna Taur

---

⭐ If you find this project useful, consider giving the repository a star!

