#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np

# Load assets
knn = joblib.load("models/agrovet_knn.pkl")
le_region = joblib.load("models/le_region.pkl")
le_specialty = joblib.load("models/le_specialty.pkl")
df = pd.read_csv("models/agrovet_data.csv")

def find_nearest_agrovet(region, specialty):
    """Find the nearest agrovet given a region and specialty."""
    try:
        region_enc = le_region.transform([region])[0]
        specialty_enc = le_specialty.transform([specialty])[0]
    except ValueError:
        return {"error": f"Unknown region '{region}' or specialty '{specialty}'."}

    query = np.array([[region_enc, specialty_enc]])
    distance, index = knn.kneighbors(query)
    row = df.iloc[index[0][0]]

    return {
        "agrovet_name": row["agrovet_name"],
        "region": row["region"],
        "specialty": row["specialty"],
        "phone": row["phone"],
        "email": row["email"],
        "statement": row["statement"],
        "image_path": row["image_path"],
        "message": f"📞 Call {row['phone']} or ✉️ Email {row['email']} for help with {row['specialty']} in {row['region']}."
    }

if __name__ == "__main__":
    res = find_nearest_agrovet("Machakos", "fish farming supplies")
    print(res)
