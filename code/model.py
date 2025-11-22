# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# config
INPUT_NAME = "train_prepared"
OUTPUT_NAME = "titanic_predictions"

# load
df = dataiku.Dataset(INPUT_NAME).get_dataframe()

# prep
cols_drop = [c for c in ["Name", "Ticket", "Cabin"] if c in df.columns]
df = df.drop(columns=cols_drop)

if "Sex" in df.columns:
    df["Sex"] = df["Sex"].map({"female": 0, "male": 1}).astype("float")

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2}).astype("float")

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

if "Fare" in df.columns:
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce").fillna(df["Fare"].median())

# target & features
assert "Survived" in df.columns, "Brak kolumny 'Survived' w zbiorze"
df = df.dropna(subset=["Survived"])
y = df["Survived"].astype(int)
X = df.drop(columns=["Survived"])

X = X.select_dtypes(include=["number"])

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# eval
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print("Confusion matrix:")
print(cm)

# save predictions
out = X_test.copy()
out["y_true"] = y_test.values
out["y_pred"] = y_pred
out["proba_survived"] = y_proba

dataiku.Dataset(OUTPUT_NAME).write_with_schema(out)




