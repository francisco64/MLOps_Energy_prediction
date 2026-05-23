# AEMO VIC1 Electricity Price Forecasting Project

![AEMO VIC1 Project - Context and Dataset](AEMO%20VIC1%20Price%20Spike%20Classification%20Project%20%E2%80%94%20Context%20and%20Dataset.png)

This project explores whether short-term electricity market conditions in Victoria can be used to forecast wholesale electricity prices in the Australian National Electricity Market (NEM).

The current modeling direction is a time-series forecasting task for the AEMO `VIC1` region: predict future regional reference price (`RRP`) values from recent price behavior, demand, generation availability, interconnector flow, and calendar features. MLflow is used to track model runs, metrics, artifacts, and parameters.

## What This Project Is Trying To Achieve

Wholesale electricity prices are volatile and are influenced by demand, available generation, network conditions, interconnector behavior, and time-based market patterns. The goal is to turn raw AEMO dispatch data into a machine learning workflow that can:

- Build a clean time-series dataset for the Victoria `VIC1` region.
- Engineer lag, rolling-window, exponential moving average, calendar, and holiday features.
- Train machine learning models to forecast future `RRP` values.
- Compare feature encoding strategies, including cyclical time features and radial basis function (RBF) time encodings.
- Track experiments with MLflow so model performance, parameters, and artifacts are reproducible.
- Provide a foundation for a future MLOps pipeline with scheduled retraining, model registry, API serving, monitoring, and observability.

## Dataset Context

The included sample data comes from AEMO public monthly archive dispatch files for January 2025:

- `test_data/PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#202501010000.CSV`
- `test_data/PUBLIC_ARCHIVE#DISPATCHREGIONSUM#FILE01#202501010000.CSV`

The notebook filters these files to:

- `REGIONID == "VIC1"`
- `RUNNO == 1`
- `INTERVENTION == 0`

The key joined fields used for forecasting include:

- `SETTLEMENTDATE`
- `DISPATCHINTERVAL`
- `REGIONID`
- `RRP`
- `TOTALDEMAND`
- `AVAILABLEGENERATION`
- `AVAILABLELOAD`
- `DEMANDFORECAST`
- `DISPATCHABLEGENERATION`
- `DISPATCHABLELOAD`
- `NETINTERCHANGE`

The project originally explored a forward-looking spike label:

```text
target_price_spike_next_60min = 1 if max(RRP over the next 12 dispatch intervals) >= 300
```

AEMO dispatch intervals are 5 minutes, so 12 intervals represent the next 60 minutes. That classification framing has now been superseded by the forecasting direction. The current experiments use `RRP` as the prediction target and evaluate forecast error over future horizons.

## Current Experiment Flow

The main work is in `Experiments.ipynb`.

The notebook currently does the following:

1. Loads AEMO dispatch price and regional summary CSV files.
2. Filters to the `VIC1` region and normal market dispatch runs.
3. Joins price and regional operational data on settlement time, dispatch interval, region, run number, and intervention flag.
4. Builds a supervised time-series forecasting matrix for `RRP`.
5. Creates time-series features using:
   - lag values: 1, 2, 3, 6, 12, 24, and 288 intervals
   - rolling mean, min, and max windows
   - exponential moving averages
   - cyclical time encodings
   - RBF time encodings
   - weekend, business-hour, and Victorian holiday indicators
6. Trains XGBoost models.
7. Logs experiments to MLflow, including:
   - datasets
   - model parameters
   - model artifacts
   - training metrics
   - forecasting metrics
   - forecast plots

## MLOps Direction

The architecture artifact in this repository outlines a broader production direction:

![MLOps Architecture](MLOps%20Architecture.png)

The intended MLOps workflow is:

- Airflow orchestrates training and scheduled workflows.
- Training code builds features, trains a model, and logs outputs to MLflow.
- MLflow tracks experiments and stores registered model artifacts.
- FastAPI serves the selected model for prediction.
- A relational database stores MLflow metadata and prediction logs.
- Monitoring jobs evaluate drift and prediction behavior.
- Prometheus and Grafana provide observability.

The current repository is still at the experimentation stage. The notebook and architecture assets document the modeling direction and the target platform design.

## Repository Contents

```text
.
├── Experiments.ipynb
├── AEMO VIC1 Price Spike Classification Project — Context and Dataset.png
├── MLOps Architecture.png
├── mlops_architecture_v2-6.html
├── test_data/
│   ├── PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#202501010000.CSV
│   └── PUBLIC_ARCHIVE#DISPATCHREGIONSUM#FILE01#202501010000.CSV
└── README.md
```

Local MLflow runtime files such as `mlflow.db` and `mlruns/` are intentionally ignored by Git because they are generated artifacts and can become very large.

## Running The Notebook

Create or activate a Python environment with the main notebook dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost mlflow holidays
```

Then open:

```bash
jupyter notebook Experiments.ipynb
```

If using MLflow locally, start the tracking UI separately:

```bash
mlflow ui --port 5001
```

The notebook is configured to use:

```text
http://localhost:5001
```

## Notes And Limitations

- The included data is a sample month, not a full historical training corpus.
- The repository still contains some naming and visual context from the earlier price-spike classification framing.
- Forecast quality should be evaluated with time-aware validation rather than random splits.
- Price spikes remain important, but in the current direction they are handled as difficult regions of the continuous `RRP` forecast rather than as the primary target label.
- `mlflow.db` and `mlruns/` are local experiment-tracking artifacts and are not committed to the repository.

## Next Steps

- Move reusable notebook logic into Python modules.
- Add a repeatable training script.
- Add a `requirements.txt` or environment file.
- Clean up remaining classification-era naming if the project is now fully forecasting-focused.
- Add forecast metrics such as MAE, RMSE, R2, and horizon-specific error summaries.
- Add time-series cross-validation across multiple months.
- Convert the architecture design into runnable services once the modeling approach is stable.
