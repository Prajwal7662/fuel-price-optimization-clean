# Fuel-price-optimization-ml
> 🚀 End-to-end Machine Learning project for fuel price optimization

⛽ Fuel Price Optimization using Machine Learning

This project implements an end-to-end Machine Learning pipeline to recommend an optimal daily fuel price for a retail fuel station in a competitive market.
The objective is to maximize profit by predicting demand and selecting the best price based on historical data, costs, and competitor prices.

📌 Problem Statement

Fuel retailers adjust prices daily while competing with nearby stations.
Choosing the wrong price can reduce sales or profit.

This system helps answer:

What should today’s fuel price be?

How much volume can be expected?

What profit can be achieved?

🧠 Solution Overview

The solution follows a simple, efficient, and business-oriented ML approach:

1️⃣ Data Ingestion

Historical fuel price and sales data is loaded from a CSV file.

2️⃣ Data Cleaning & Feature Engineering

Handles missing values

Creates useful features such as:

Average competitor price

Price difference vs competitors

Day-based patterns

3️⃣ Machine Learning Model

Random Forest Regressor

Predicts expected fuel demand (volume)

Model performance evaluated using MAE

4️⃣ Price Optimization Logic

Simulates multiple price options

Predicts demand for each price

Calculates profit using:

Profit = (Price − Cost) × Predicted Volume


Selects the price that maximizes profit

5️⃣ Final Output

Recommended price

Expected sales volume

Expected profit

📂 Project Structure

fuel-price-optimization-clean/

│

├── fuel_price_optimization.py

├── README.md

├── requirements.txt

├── today_example.json

├── .gitignore

│

└── data/

    └── raw/

        └── oil_retail_history.csv

▶️ How to Run the Project

Step 1: Install dependencies

pip install -r requirements.txt


Step 2: Run the pipeline

python fuel_price_optimization.py


📈 Sample Output
{
  "recommended_price": 105.9,
  "expected_volume": 13117,
  "expected_profit": 129861.22
}

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Random Forest Regression

📊 Dataset Note

This repository contains a representative sample of the dataset to keep the project lightweight.
The full dataset can be shared upon request.

🚀 Key Highlights

Clean and readable code

End-to-end ML pipeline

Business-focused optimization

Easy to extend for real-world deployment