"""
PURPOSE:
Establishes the foundational Machine Learning pipeline for Triage AI. Ingests raw 
Electronic Health Records (EHR) from the UCI 130-US Hospitals dataset, enforces data 
cleaning and clinical filtering rules, and trains an ensemble classifier to predict 
30-day hospital readmission risk.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ==============================================================================
# 1. ISSUE #1: DATA INGESTION & CLINICAL HYGIENE
# ==============================================================================
print("=" * 65)
print("  TRIAGE AI: BASELINE 30-DAY READMISSION CLASSIFIER")
print("=" * 65)
print("\n[Step 1/4] Loading and cleaning raw dataset...")

# Handle relative path whether run from root or inside src/
data_path = 'data/diabetic_data.csv'
if not os.path.exists(data_path):
    data_path = '../data/diabetic_data.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError("Could not find 'diabetic_data.csv' in 'data/' or '../data/' folder.")

# Load raw records, treating '?' as missing (NaN)
df = pd.read_csv(data_path, na_values='?')
print(f"  -> Raw records ingested: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop unique identifier keys and columns with excessive missingness (>40%)
cols_to_drop = ['encounter_id', 'weight', 'payer_code', 'medical_specialty']
df = df.drop(columns=cols_to_drop)

# Keep only the first encounter per patient to avoid data leakage
df = df.drop_duplicates(subset=['patient_nbr'], keep='first').drop(columns=['patient_nbr'])

# Filter out terminal / hospice cases (clinical impossibilities for readmission)
expired_ids = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(expired_ids)]

# Binary target: 1 = Readmitted within 30 days (<30), 0 = Not readmitted early
df['target_readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
df = df.drop(columns=['readmitted'])

positive_count = df['target_readmitted'].sum()
total_count = len(df)
print(f"  -> Cleaned cohort: {total_count} patients")
print(f"  -> Class distribution: {positive_count} readmitted ({positive_count/total_count*100:.2f}%), {total_count - positive_count} safe")

# ==============================================================================
# 2. FEATURE SELECTION & ENCODING
# ==============================================================================
print("\n[Step 2/4] Selecting features and encoding categories...")

num_features = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
    'num_medications', 'number_outpatient', 'number_emergency', 
    'number_inpatient', 'number_diagnoses'
]

cat_features = ['race', 'gender', 'age', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']

# One-hot encode categorical features (drop_first avoids multicollinearity)
df_encoded = pd.get_dummies(df[num_features + cat_features], drop_first=True)
y = df['target_readmitted']
print(f"  -> Engineered feature matrix: {df_encoded.shape[1]} input variables")

# ==============================================================================
# 3. ISSUE #2: STRATIFIED SPLIT & BASELINE MODEL TRAINING
# ==============================================================================
print("\n[Step 3/4] Performing 80/20 stratified split & training Random Forest...")

X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest with balanced class weights to address 10:1 class imbalance
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 4. EVALUATION & METRICS GENERATION
# ==============================================================================
print("\n[Step 4/4] Evaluating model performance on unseen test data...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

roc_score = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 65)
print("                     CLINICAL MODEL RESULTS")
print("=" * 65)
print(f"ROC-AUC Score: {roc_score:.4f}")
print("\nConfusion Matrix (Test Set):")
print(f"  True Negatives  (Safe correctly predicted)       : {cm[0][0]}")
print(f"  False Positives (Safe flagged as high-risk)      : {cm[0][1]}")
print(f"  False Negatives (Missed high-risk patients)      : {cm[1][0]}")
print(f"  True Positives  (High-risk correctly intercepted): {cm[1][1]}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No / >30 Days (Safe)', 'Readmitted <30 Days (High-Risk)']))
print("=" * 65)
print("[STATUS] Baseline Machine Learning pipeline completed successfully.")