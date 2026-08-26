# XAI-Based House Price Prediction System

## Project Overview
This mini project predicts California house values using a Random Forest Regressor and explains the model using Feature Importance, SHAP and Partial Dependence Plot (PDP).

## Dataset
The project uses the **California Housing dataset** supplied with this repository. It contains 20,640 observations, eight numeric predictive variables, the categorical `ocean_proximity` variable, and `median_house_value` as the target.

## Workflow
Dataset → preprocessing → train/test split → Random Forest → evaluation → Feature Importance → SHAP global/local explanations → PDP → Responsible AI analysis.

## XAI Techniques
- Random Forest Feature Importance
- SHAP Summary Plot
- SHAP Waterfall Plot
- Partial Dependence Plot

## Results from the current run
- MAE: **33940.25**
- RMSE: **51238.93**
- R²: **0.7996**

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/train_and_explain.py
```

## Output files
- `outputs/metrics.txt`
- `outputs/feature_importance.png`
- `outputs/shap_summary.png`
- `outputs/shap_waterfall.png`
- `outputs/pdp_median_income.png`
- `outputs/sample_predictions.csv`

## Responsible AI
The model is an educational demonstration. SHAP explains model behavior, not causation. Housing predictions can reflect historical and geographic biases, so real deployment requires representative current data, validation, monitoring and human oversight.
