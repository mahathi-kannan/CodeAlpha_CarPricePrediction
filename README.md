# CodeAlpha Internship - Car Price Prediction with Machine Learning

This repository contains the complete source code and documentation for **Task 3: Car Price Prediction**, completed during my Data Science Internship at **CodeAlpha**.

## 📌 Project Overview
The objective of this project is to build a predictive machine learning model that estimates the selling price of used cars based on various features such as current showroom price, kilometers driven, fuel type, transmission, and the age of the vehicle. 

Predicting vehicle valuation accurately is a core real-world application of machine learning in the automotive and e-commerce industries.

## 📊 Dataset Features
The dataset used for this project contains the following parameters:
* `Car_Name`: The model name of the vehicle (Dropped during preprocessing to avoid overfitting).
* `Year`: Manufacturing year (Used to calculate `Car_Age`).
* `Present_Price`: Current showroom price of the car.
* `Driven_kms`: Total kilometers the car has been driven.
* `Fuel_Type`: Fuel type of the vehicle (Petrol, Diesel, CNG).
* `Selling_type`: Sold by a Dealer or an Individual.
* `Transmission`: Gearbox type (Manual, Automatic).
* `Owner`: Number of previous owners.
* **`Selling_Price` (Target Variable)**: The price at which the car is being sold.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries Used:** * `Pandas` & `NumPy` (Data Manipulation)
  * `Matplotlib` & `Seaborn` (Data Visualization)
  * `Scikit-Learn` (Model Training & Evaluation)

## 🚀 Machine Learning Pipeline
1. **Data Cleaning:** Checked for missing data and unexpected null values.
2. **Feature Engineering:** Extracted a new feature `Car_Age` by subtracting the manufacturing year from the current year to capture depreciation trends.
3. **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) to process non-numeric features like `Fuel_Type` and `Transmission`.
4. **Data Splitting:** Divided the data into an 80% training set and a 20% testing set.
5. **Model Implementation:** Trained a **Random Forest Regressor** to handle non-linear relationships within the data.
6. **Evaluation:** Evaluated the model using Mean Absolute Error (MAE) and the $R^2$ (R-squared) score.

## 📈 Results & Visualizations
The model successfully maps features like high showroom price and lower vehicle age to a higher predicted selling price. A scatter plot comparing the *Actual Prices vs. Predicted Prices* is generated at the end of the script to visually confirm model alignment.

