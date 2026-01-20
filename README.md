 # Coronary Artery Disease (CAD) Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)

A Data Science project and machine learning model that predicts CAD disease risk with 90% AUC accuracy, based on the most important predictors

## What is Coronary Artery Disease?

Coronary Artery Disease (CAD) is a common heart disease affecting the main blood supply to the heart, namely the coronary arteries. A buildup of fats, aswell as cholesterol and other substances, narrows the artery walls, causing CAD. 

### Symptoms:

1. Angina (Chest pain) - Squeezing, pressure and heaviness in the chest.
2. A shortness of breath
3. Fatigue, since your heart cannot pump enough blood to meet your body's needs.

### Interesting statistics regarding CAD:

1. CAD accounts for 610 000 deaths (1 in 4) annually.
2. 1 in 4 deaths in the United States is caused by CAD.
3. 1 in 6 deaths in South Africa caused by CAD.

## Key Model Features:

| # | Feature | Medical Meaning | Impact |
|---|---------|----------------|--------|
| 1 | **Thallium Stress Level** | Poor blood flow to muscles during exercise. | **Highest** |
| 2 | **Number of calcified vessels** | Number of major vessels visible under fluoroscopy.| **Strong** |
| 3 | **Exercise Angina** | Chest pain caused by physical exertion. | **Strong** |


## What does my Machine Learning model present?
- **92% AUC score** - This score performs better than the ECG standard (60-70% prediction rate)
- **Interactive Mini Web Application** - User-friendly Streamlit interface
- **Comprehensive Project Structure and Analysis** - Full data science workflow documented, from data cleaning to model discussion.

## Using the model:

### Prerequisites:
- Python [e.g., 3.8 or higher]
- pip package manager

### Installation:
```bash
# Clone the repository
git clone https://github.com/Ryan-Naidoo-05/Heart-Disease-Predictor-.git
cd Heart-Disease-Predictor-

# Installing the dependencies<img width="1700" height="1424" alt="ROC_AUC" src="https://github.com/user-attachments/assets/f0498cf0-5317-4fdb-be08-aacbfcbe59cf" />

pip install -r requirements.txt

# Run the Jupyter Notebook and the
# Use the cells to visualise plots and inspect the logistic regression model that was fit to
# make the classifications.


# Use the live demonstration, built on the Machine Learning model, via Streamlit.
# In the terminal, run:
streamlit run app.py

```
## Model Results and Evaluation:

The model achieves an AUC score of 92.1% by taking 0.40 as our cutoff probability. I adjusted this cutoff to ensure that the model minimises false negatives. In a medical scenario, we definitely want to minimise the number of false negatives, seeing as mistakes can cause fatal errors resulting in death for the patients, whereas false alarms (while still tedious) do not cause the loss of lives. I used a logistic regression model from statsmodels for the classification, taking the 3 most important features as our predictors. 

<img width="1700" height="1424" alt="ROC_AUC" src="https://github.com/user-attachments/assets/fdfbc8ab-3342-4ffa-8432-76935cbf500a" />

# Improvement areas:

1. Adjusting the cutoff to try to minimise more false positives, since I mainly prioritised reducing false negatives at all costs.
2. Incorporating more predictors into the algorithm. Three predictors tied for being the most correlated with heart disease, but I selectively chose only one that seemed most relevant to me.
3. Changing the model type to more commonly-used ones in the industry (and more powerful), such as XGBoost or LightGBM. I stuck with a Logistic Regression from statsmodels, since I wanted to incorporate skills taught to me in my Data Science class.
