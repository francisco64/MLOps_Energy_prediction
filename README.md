# AEMO VIC1 Price Spike Classification Project

![AEMO VIC1 Price Spike Classification Project - Context and Dataset](AEMO%20VIC1%20Price%20Spike%20Classification%20Project%20%E2%80%94%20Context%20and%20Dataset.png)

This project explores whether short-term electricity market conditions in Victoria can be used to detect or anticipate wholesale electricity price spikes in the Australian National Electricity Market (NEM).

The initial framing is a binary classification task for the AEMO `VIC1` region: predict whether the regional reference price (`RRP`) will reach a spike threshold in the next 60 minutes. The notebook also extends the work into a forecasting experiment, where XGBoost models predict future `RRP` values and MLflow tracks model runs, metrics, artifacts, and parameters.

## What This Project Is Trying To Achieve

Electricity price spikes are operationally important because they can indicate stress in the grid, demand-supply imbalance, transmission constraints, generator outages, or unusual market behavior. The goal is to turn raw AEMO dispatch data into a machine learning workflow that can:

- Build a clean time-series dataset for the Victoria `VIC1` region.
- Engineer lag, rolling-window, exponential moving average, calendar, and holiday features.
- Define a forward-looking price spike target.
- Train baseline machine learning models to classify or forecast price behavior.
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

The key joined fields include:

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

In the current notebook, the classification target is:

```text
target_price_spike_next_60min = 1 if max(RRP over the next 12 dispatch intervals) >= 300
```

AEMO dispatch intervals are 5 minutes, so 12 intervals represent the next 60 minutes. In the included January 2025 sample, the processed `VIC1` classification dataset contains 8,916 rows, with 53 positive spike-window examples. That makes the target highly imbalanced, which is an important modeling constraint.

## Current Experiment Flow

The main work is in `Experiments.ipynb`.

The notebook currently does the following:

1. Loads AEMO dispatch price and regional summary CSV files.
2. Filters to the `VIC1` region and normal market dispatch runs.
3. Joins price and regional operational data on settlement time, dispatch interval, region, run number, and intervention flag.
4. Builds the future price spike label for the next 60 minutes.
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
- The classification target is very imbalanced, so accuracy alone is not a useful evaluation metric.
- The notebook currently mixes classification framing with forecasting experiments. This is expected at the exploration stage, but future production code should separate these into distinct pipelines.
- `mlflow.db` and `mlruns/` are local experiment-tracking artifacts and are not committed to the repository.

## Next Steps

- Move reusable notebook logic into Python modules.
- Add a repeatable training script.
- Add a `requirements.txt` or environment file.
- Separate classification and forecasting experiments.
- Add imbalance-aware classification metrics such as precision, recall, F1, PR-AUC, and confusion matrices.
- Add time-series cross-validation across multiple months.
- Convert the architecture design into runnable services once the modeling approach is stable.
