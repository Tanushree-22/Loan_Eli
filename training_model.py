# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv('Loan_Data (1).csv')

# Clean and engineer features
df['Dependents'] = df['Dependents'].replace('3+', '3').fillna('0').astype(int)
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = df['LoanAmount'] * 0.08 * (1 + 0.08)**df['Loan_Amount_Term'] / ((1 + 0.08)**df['Loan_Amount_Term'] - 1)
df['Income_to_EMI'] = df['Total_Income'] / df['EMI']
df['EMI'] = df['EMI'].fillna(df['EMI'].median())
df['Income_to_EMI'] = df['Income_to_EMI'].fillna(df['Income_to_EMI'].median())

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Preprocessing
categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
numerical = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
             'Loan_Amount_Term', 'Total_Income', 'EMI', 'Income_to_EMI', 'Credit_History']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler())]), numerical),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, 'loan_model_pipeline.pkl')
print("Model pipeline saved to loan_model_pipeline.pkl")
