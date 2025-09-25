# Simple Linear Regression - House Price Prediction

## Overview

This project implements a **Simple Linear Regression** model to predict house prices using the 'area' feature from the `Housing.csv` dataset. This was completed as part of a machine learning assignment (Task 3).

## Setup & Run

1.  **Dependencies:** Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

2.  **Run:** Execute the main script from your terminal:
    ```bash
    python linear_regression.py
    ```

## Key Results (Example)

The script trains the model ($\text{Price} = (m \times \text{Area}) + c$) and evaluates performance on test data:

| Metric | Example Value |
| :--- | :--- |
| **R-squared ($R^2$)** | 0.6543 |
| **MAE** | $876,543.21 |
| **Area Coefficient ($m$)** | $456.78 |

*Interpretation: The model explains roughly 65% of the price variance. For every 1 unit increase in area, the price increases by $456.78.*

## Future Work

* Implement **Multiple Linear Regression** by adding features like `stories` and `bathrooms`.
* Handle categorical variables (e.g., `furnishingstatus`) using One-Hot Encoding.