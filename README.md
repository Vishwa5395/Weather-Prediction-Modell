#  Weather Forecasting using Ridge Regression (Temperature Prediction)

This project aims to predict the **next day's average temperature** using historical weather data from Kolkata. We use **Ridge Regression**, a linear model that incorporates L2 regularization, to enhance prediction stability.

---

##  Overview

The workflow includes:
- Cleaning and preprocessing a real-world dataset
- Feature engineering using time-based rolling and expanding averages
- Model training using Ridge Regression
- Backtesting to evaluate the model’s performance
- Forecasting the next day’s average temperature

---

##  Dataset

- **Source**: User-provided cleaned dataset (approximately 14,000 rows)
- **Region**: Calcutta region specific dataset used for this model
- **File**: `weather_data.csv`
- **Index**: `DATE` (in format `YYYY-DD-MM`)
- **Target Variable**: `tavg` (Average Temperature in °F)

---

##  Libraries Used

- `pandas`: for data manipulation and preprocessing
- `sklearn.linear_model.Ridge`: for training the Ridge Regression model
- `sklearn.metrics.mean_absolute_error`: for model evaluation
- `datetime`: to handle time shifts and forecasting

---

## Data Preprocessing

1. **Date Parsing**:
   - Date is in `YYYY-DD-MM` format.
   - Parsed using:
     ```python
     pd.to_datetime(weather.index, format='%Y-%d-%m')
     ```

2. **Handling Missing Data**:
   - Forward fill (`ffill()`) used for handling missing values
   - Columns with more than 30% missing values were dropped

3. **Target Column**:
   - `target = tavg.shift(-1)` to predict the *next day’s* temperature

---

## Feature Engineering

1. **Rolling Averages**:
   - 3-day and 14-day rolling averages computed for `tavg`, `tmin`, and `tmax`
   - Percent difference from rolling mean added as additional features

2. **Expanding (Cumulative) Averages**:
   - Monthly and day-of-year expanding averages added:
     ```python
     weather.groupby(weather.index.month)[col].transform(expanding_mean)
     ```

3. **Final Cleanup**:
   - Dropped the first 14 rows (due to rolling window)
   - Filled any remaining `NaN` values with 0

---

## Model Training

- **Model Used**: `Ridge Regression` (`sklearn.linear_model.Ridge`)
- **Alpha**: `0.1` (regularization strength)
- **Training Data**: All preprocessed data with engineered features
- **Predictors**: All columns except `target`, `station`, and `name`

---

## Backtesting

To evaluate model robustness:
- Train-Test splits were created incrementally using a sliding window
- Step size: 90 days
- Starting after day 3650 (10th year onwards)
- **Metric Used**: Mean Absolute Error (MAE)

---

## Forecasting Next Day's Temperature

1. Train the Ridge model on the full dataset
2. Use the last available row's feature values
3. Predict the temperature for the next day (`DATE + 1 day`)
4. Display the forecast:
   ```python
   Forecast for: June 06, 2025
   Predicted Avg Temperature: 82.45 °F
