#!/usr/bin/env python3
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from preprocess import preprocess_data
from pathlib import Path

def train_model():
    df = preprocess_data("data/agrovets_regions.csv")

    # Use encoded features for similarity
    X = df[['region_enc', 'specialty_enc']]

    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(X)

    # Save model and data
    Path("models").mkdir(exist_ok=True)
    joblib.dump(knn, "models/agrovet_knn.pkl")
    df.to_csv("models/agrovet_data.csv", index=False)

    print("✅ KNN model trained and saved successfully.")

if __name__ == "__main__":
    train_model()
