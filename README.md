# Heart Disease Prediction System

End-to-end machine learning system (data, training, testing, deployment) for **AIC354 Machine Learning Fundamentals**, Assignment 2, Task 2.

| | |
|---|---|
| **Student** | Haziq Mahmood |
| **Roll number** | FA24-BSE-049 |
| **Instructor** | Dr. Rao Muhammad Adeel Nawab |
| **Institution** | COMSATS University Islamabad, Lahore Campus, Fall 2026 |
| **Live demo** | https://YOUR-APP-NAME.streamlit.app/  *(replace after deploying)* |

## What the system does

| | |
|---|---|
| **Inputs (4 attributes)** | `cp` chest pain type (1 to 4), `thalach` maximum heart rate achieved (71 to 202 bpm), `oldpeak` ST depression (0.0 to 6.2), `exang` exercise-induced angina (0 or 1) |
| **Output (1 attribute)** | `target`: **0** = no heart disease, **1** = heart disease present |
| **Model** | `StandardScaler` + Support Vector Classifier (`C=0.5`, `gamma=0.01`, RBF kernel) in a scikit-learn `Pipeline` |
| **Data** | UCI Heart Disease, Cleveland Clinic Foundation subset, 303 patients (80% train / 20% test, stratified, seed 42) |
| **Results** | Cross-validated accuracy 0.777 (training set); test accuracy 0.820, precision 0.870, recall 0.714, ROC-AUC 0.925 (61 unseen patients) |

The label-encoding step of the Titanic notebook was removed, because every attribute in this dataset is already numeric.

## Files

```
heartDiseasePredictionSystem/
├── Heart_Disease_Prediction.ipynb   complete notebook (load, EDA, preprocessing, training, evaluation, prediction)
├── processed_cleveland.csv          dataset used to train and test the model
├── heart_disease_model.pkl          trained model (loads and predicts without re-training)
├── Heart_Disease_Report.pdf         the executed notebook exported as PDF
├── app.py                           Streamlit web app (deployment)
├── model_info.json                  inputs, valid ranges, hyperparameters, metrics
├── requirements.txt                 packages needed by the deployed app
├── requirements-notebook.txt        packages needed to run the notebook (adds Jupyter, matplotlib, seaborn)
└── README.md
```

## Run the notebook

```
python -m venv venv
venv\Scripts\activate              # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements-notebook.txt
jupyter notebook
```

Open `Heart_Disease_Prediction.ipynb` and choose **Kernel > Restart & Run All**.

## Run the web app locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a **public GitHub repository** and upload the contents of this folder (`app.py` and `requirements.txt` must be at the top level of the repository).
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Select the repository, branch `main` and main file path `app.py`. In *Advanced settings* choose **Python 3.12**.
4. Click **Deploy**. When the app is live, copy its URL.
5. Paste the URL into the **first cell of the notebook** and into the table above, then re-export the notebook as PDF.
6. Test the live app: (cp 2, thalach 170, oldpeak 0.0, exang 0) must give **0**; (cp 4, thalach 110, oldpeak 2.5, exang 1) must give **1**.

`heart_disease_model.pkl` was trained with **scikit-learn 1.8.0**, which is pinned in `requirements.txt`. If the saved model cannot be loaded in the cloud, `app.py` automatically re-creates the identical model from `processed_cleveland.csv`.

## Disclaimer

Educational project. The system is **not** a medical device and must not be used to diagnose or exclude heart disease.

## Data source

Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. *Heart Disease* dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/45/heart+disease. Donor: David W. Aha. Principal investigators: Andras Janosi (Hungarian Institute of Cardiology, Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel), Robert Detrano (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation).
