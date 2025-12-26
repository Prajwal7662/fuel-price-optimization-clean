
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------
# 1. Load and clean historical data
# --------------------------------------------------

def load_and_clean_data(csv_path):
    """
    Load historical fuel data and perform basic cleaning
    """
    df = pd.read_csv(csv_path)

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Remove rows with missing or negative values
    df = df.dropna()
    df = df[(df['price'] >= 0) & (df['volume'] >= 0)]

    # Sort by date (important for lag features)
    df = df.sort_values('date').reset_index(drop=True)
    return df


# --------------------------------------------------
# 2. Feature Engineering
# --------------------------------------------------

def create_features(df):
    """
    Create simple and meaningful features
    """
    # Average competitor price
    df['comp_avg_price'] = df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)

    # Price difference vs competitors
    df['price_diff'] = df['price'] - df['comp_avg_price']

    # Day based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Lag features (previous day)
    df['price_lag_1'] = df['price'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)

    # Remove rows with NaN values caused by lagging
    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------
# 3. Train Machine Learning Model
# --------------------------------------------------

def train_model(df):
    """
    Train Random Forest model to predict demand (volume)
    """
    features = [
        'price', 'cost', 'comp_avg_price', 'price_diff',
        'day_of_week', 'is_weekend', 'price_lag_1', 'volume_lag_1'
    ]

    X = df[features]
    y = df['volume']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Simple evaluation
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"Model trained successfully | MAE: {mae:.2f}")

    return model, features


# --------------------------------------------------
# 4. Price Optimization Logic
# --------------------------------------------------

def recommend_price(model, features, today_data, last_row):
    """
    Try different prices and select the one with max profit
    Profit = (price - cost) * predicted_volume
    """
    comp_avg = np.mean([
        today_data['comp1_price'],
        today_data['comp2_price'],
        today_data['comp3_price']
    ])

    # Generate price candidates around competitor price
    price_candidates = np.arange(comp_avg - 3, comp_avg + 3, 0.1)

    best_profit = -np.inf
    best_price = today_data['price']
    best_volume = 0

    for price in price_candidates:
        # Business rule: price should be higher than cost
        if price <= today_data['cost']:
            continue

        # Prepare input row
        row = pd.DataFrame([{
            'price': price,
            'cost': today_data['cost'],
            'comp_avg_price': comp_avg,
            'price_diff': price - comp_avg,
            'day_of_week': pd.to_datetime(today_data['date']).dayofweek,
            'is_weekend': int(pd.to_datetime(today_data['date']).dayofweek >= 5),
            'price_lag_1': last_row['price'],
            'volume_lag_1': last_row['volume']
        }])

        # Predict demand
        predicted_volume = model.predict(row[features])[0]

        # Calculate profit
        profit = (price - today_data['cost']) * predicted_volume

        if profit > best_profit:
            best_profit = profit
            best_price = price
            best_volume = predicted_volume

    return {
        'recommended_price': round(best_price, 2),
        'expected_volume': int(best_volume),
        'expected_profit': round(best_profit, 2)
    }


# --------------------------------------------------
# 5. End-to-End Execution
# --------------------------------------------------

def run_pipeline(history_csv, today_json):
    # Load historical data
    df = load_and_clean_data(history_csv)

    # Feature engineering
    df = create_features(df)

    # Train model
    model, features = train_model(df)

    # Load today's data
    with open(today_json, 'r') as f:
        today_data = json.load(f)

    # Use last available row for lag values
    last_row = df.iloc[-1]

    # Recommend price
    result = recommend_price(model, features, today_data, last_row)

    print("\nFinal Recommendation:")
    print(json.dumps(result, indent=2))


# --------------------------------------------------
# Example Run
# --------------------------------------------------
run_pipeline('data/raw/oil_retail_history.csv', 'today_example.json')
