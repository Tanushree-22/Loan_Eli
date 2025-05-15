import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("Loan_Data (1).csv")

# Preprocess
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

# Encode target
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Load saved model
pipeline = joblib.load("loan_model_pipeline.pkl")

# Predict and evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.4f}")
