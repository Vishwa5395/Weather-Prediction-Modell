# import pandas as pd
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error

# # === Step 1: Load and Preprocess the Data ===
# weather = pd.read_csv('weather_data.csv', index_col='DATE')
# weather.index = pd.to_datetime(weather.index)
# weather.columns = weather.columns.str.lower()

# # Fill forward and drop sparse columns
# weather = weather.ffill()
# null_pct = weather.isnull().mean()
# weather = weather.loc[:, null_pct < 0.3]

# # Create 'target' column: tomorrow's temperature
# weather['target'] = weather['tavg'].shift(-1)
# weather = weather.ffill()

# # === Step 2: Feature Engineering ===
# def pct_diff(old, new):
#     return (new - old) / old

# def compute_rolling(df, horizon, col):
#     label = f"rolling_{horizon}_{col}"
#     df[label] = df[col].rolling(horizon).mean()
#     df[f"{label}_pct"] = pct_diff(df[col], df[label])
#     return df

# # Rolling averages for 3 and 14 days
# for horizon in [3, 14]:
#     for col in ['tavg', 'tmin', 'tmax']:
#         if col in weather.columns:
#             weather = compute_rolling(weather, horizon, col)

# # Expanding (monthly and day-of-year) averages
# def expand_mean(df):
#     return df.expanding(1).mean()

# for col in ['tavg', 'tmin', 'tmax']:
#     if col in weather.columns:
#         weather[f'month_avg_{col}'] = weather.groupby(weather.index.month)[col].transform(expand_mean)
#         weather[f'day_avg_{col}'] = weather.groupby(weather.index.day_of_year)[col].transform(expand_mean)

# # Clean up
# weather = weather.iloc[14:].copy()
# weather = weather.fillna(0)

# # === Step 3: Model Training ===
# predictors = weather.columns.difference(['target', 'name', 'station'])
# model = Ridge(alpha=0.1)
# model.fit(weather[predictors], weather['target'])

# # === Step 4: Predict Tomorrow's Temperature ===
# latest_row = weather.iloc[-1:].copy()
# prediction = model.predict(latest_row[predictors])[0]


# print(f" Forecast Date: {weather.index[-1] + pd.Timedelta(days=1):%B %d, %Y}")
# print(f" Predicted Avg Temperature: {prediction:.2f} °F")


import pandas as pd

weather = pd.read_csv('weather_data.csv', index_col='DATE')
weather.index = pd.to_datetime(weather.index)
weather.columns = weather.columns.str.lower()


weather = weather.ffill() # Filling missing values first

null_pct = weather.isnull().mean() #Dropping very sparse columns AFTER ffill

weather = weather.loc[:, null_pct < 0.3]  #Threshold is 30%
weather['target'] = weather['tavg'].shift(-1)
weather = weather.ffill()
def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(df, horizon, col):
    label = f"rolling_{horizon}_{col}"
    df[label] = df[col].rolling(horizon).mean()
    print(df)
    df[f"{label}_pct"] = pct_diff(df[col], df[label])
    return df

for horizon in [3, 14]:
    for col in ['tavg', 'tmin', 'tmax']:
        if col in weather.columns:
            weather = compute_rolling(weather, horizon, col)
def expand_mean(df):
    return df.expanding(1).mean()

for col in ['tavg', 'tmin', 'tmax']:
    if col in weather.columns:
        weather[f'month_avg_{col}'] = weather.groupby(weather.index.month)[col].transform(expand_mean)
        weather[f'day_avg_{col}'] = weather.groupby(weather.index.day_of_year)[col].transform(expand_mean)


print(weather)

weather = weather.iloc[14:].copy()
weather = weather.fillna(0)  # Fill remaining

predictors = weather.columns.difference(['target', 'name', 'station'])
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

model = Ridge(alpha=0.1)

def backtesting(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i]
        test = weather.iloc[i:i+step]

        model.fit(train[predictors], train['target'])
        preds = pd.Series(model.predict(test[predictors]), index=test.index)

        combined = pd.concat([test['target'], preds], axis=1)
        combined.columns = ['actual', 'prediction']
        combined['diff'] = (combined['actual'] - combined['prediction']).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)

predictions = backtesting(weather, model, predictors)
print("MAE:", mean_absolute_error(predictions['actual'], predictions['prediction']))


# Prediction of Next Day's Temperature
# Retraining model on full dataset
model.fit(weather[predictors], weather['target'])



# Select last available row for forecasting
latest_row = weather.iloc[-1:].copy()
# print(pd.Timedelta(days=1)+latest_row.index[0])
next_day = latest_row.index[0] + pd.Timedelta(days=1)

# print(next_day)

predicted_temp = model.predict(latest_row[predictors])[0]

print(f"\n Forecast for: {next_day.strftime('%B %d, %Y')}")
print(f" Predicted Avg Temperature: {predicted_temp:.2f} °F")
