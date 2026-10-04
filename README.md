# 🚲 BoomBikes Bike-Sharing Demand Prediction (in progress)

Predicting daily bike rental demand for BoomBikes using multiple linear regression, built on weather and calendar data.

**Status:** in progress. My first version was AI-assisted — I'm currently rebuilding it by hand, step by step, to fully understand the data workflow and what drives the model's results, rather than just the final output.

## What it does

`boombikes_regression.py`:

1. Loads a bike-sharing dataset (daily records of season, weather, and ride counts)
2. Explores it — missing values, summary statistics, correlation with demand
3. Trains a multiple linear regression model to predict daily ride count (`cnt`)
4. Reports R², RMSE, and which features matter most

## Expected data

A CSV (commonly `day.csv`) with columns similar to the classic UCI/Capital Bikeshare dataset:

`season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt`

## Run it

```bash
pip install pandas scikit-learn
python boombikes_regression.py path/to/day.csv
```

## Next steps

- Add more feature engineering (interaction terms, categorical encoding)
- Compare linear regression against regularized models (Ridge/Lasso)
- Add train/test visualizations
