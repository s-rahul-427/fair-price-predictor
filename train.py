import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "flipkart_com-ecommerce_sample.csv"
MODEL_PATH = "model.pkl"

def parse_main_category(tree_str):
    if pd.isna(tree_str):
        return "Unknown"
    try:
        clean_str = re.sub(r'["\[\]]', '', str(tree_str))
        parts = clean_str.split(">>")
        return parts[0].strip() if parts else "Unknown"
    except Exception:
        return "Unknown"

def parse_rating(rating_str):
    if pd.isna(rating_str):
        return 3.5
    try:
        match = re.search(r"(\d+(\.\d+)?)", str(rating_str))
        if match:
            val = float(match.group(1))
            return val if 1.0 <= val <= 5.0 else 3.5
    except Exception:
        pass
    return 3.5

def train_model():
    print("Loading Kaggle Flipkart dataset...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cannot find {CSV_PATH} in root directory!")

    df = pd.read_csv(CSV_PATH)
    
    # Preprocessing
    df['retail_price'] = pd.to_numeric(df['retail_price'], errors='coerce')
    df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
    df = df.dropna(subset=['retail_price', 'discounted_price'])
    df = df[df['retail_price'] > 0]
    
    df['main_category'] = df['product_category_tree'].apply(parse_main_category)
    df['brand_clean'] = df['brand'].fillna("Generic").astype(str)
    df['rating_clean'] = df['overall_rating'].apply(parse_rating)
    df['discount_pct'] = ((df['retail_price'] - df['discounted_price']) / df['retail_price']) * 100
    df['discount_pct'] = df['discount_pct'].clip(0, 100)

    # Features and Target
    X = df[['retail_price', 'main_category', 'brand_clean', 'rating_clean', 'discount_pct']]
    y = df['discounted_price']

    # Preprocessing Pipeline
    categorical_features = ['main_category', 'brand_clean']
    numerical_features = ['retail_price', 'rating_clean', 'discount_pct']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=50), categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor model...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Training Complete! R2 Score: {r2:.4f}, MAE: {mae:.2f}")

    # Save model along with baseline dataframe subset for comparison features
    sample_baselines = df[['product_name', 'main_category', 'brand_clean', 'retail_price', 'discounted_price', 'rating_clean']].sample(n=min(2000, len(df)), random_state=42)
    
    artifact = {
        'pipeline': model_pipeline,
        'r2_score': r2,
        'mae': mae,
        'baselines': sample_baselines
    }
    
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved successfully to '{MODEL_PATH}'!")

if __name__ == "__main__":
    train_model()