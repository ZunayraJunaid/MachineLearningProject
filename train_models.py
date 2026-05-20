# train_models.py
# Run this file once to train and save all models

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# --- Load dataset manually ---
with open('dataset/chronic_kidney_disease.arff', 'r') as f:
    lines = f.readlines()

columns = []
in_data = False
data_rows = []

for line in lines:
    line_stripped = line.strip()

    if line_stripped == '' or line_stripped.startswith('%'):
        continue

    if line_stripped.lower().startswith('@attribute'):
        parts = line_stripped.split()
        col_name = parts[1].strip("'")
        columns.append(col_name)

    elif line_stripped.lower() == '@data':
        in_data = True

    elif in_data:
        parts = [p.strip() for p in line_stripped.split(',')]
        parts = [np.nan if p == '' else p for p in parts]
        if len(parts) == len(columns):
            data_rows.append(parts)

df = pd.DataFrame(data_rows, columns=columns)
df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# --- Convert ---
for col in ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'sg', 'al', 'su']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

binary_map = {'normal': 0, 'abnormal': 1, 'present': 1, 'notpresent': 0,
              'yes': 1, 'no': 0, 'good': 1, 'poor': 0}
for col in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
    df[col] = df[col].map(binary_map)

df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})

X = df.drop('class', axis=1)
y = df['class']

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Define Models ---
models = {
    'random_forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boosting'  : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'adaboost'           : AdaBoostClassifier(n_estimators=100, random_state=42),
    'voting_classifier'  : VotingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
        ], voting='hard'),
    'stacking_classifier': StackingClassifier(
        estimators=[
            ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
        ], final_estimator=LogisticRegression(random_state=42), cv=5)
}

# --- Train and Save ---
os.makedirs('models', exist_ok=True)
os.makedirs('preprocessing', exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...", end=' ')
    model.fit(X_train, y_train)
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved")

with open('preprocessing/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('preprocessing/selected_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nAll models and preprocessing objects saved successfully")