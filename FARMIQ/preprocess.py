#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

def preprocess_data(csv_path="data/agrovets_regions.csv"):
    df = pd.read_csv(csv_path)

    # Encode categorical data
    le_region = LabelEncoder()
    le_specialty = LabelEncoder()

    df['region_enc'] = le_region.fit_transform(df['region'])
    df['specialty_enc'] = le_specialty.fit_transform(df['specialty'])

    # Save encoders
    Path("models").mkdir(exist_ok=True)
    joblib.dump(le_region, "models/le_region.pkl")
    joblib.dump(le_specialty, "models/le_specialty.pkl")

    # Save processed data
    df.to_csv("data/agrovets_processed.csv", index=False)
    print("✅ Preprocessing complete — encoded data saved.")
    return df

if __name__ == "__main__":
    preprocess_data("data/agrovets_regions.csv")
