# XAI-Based House Price Prediction System

## 1. Abstract
This mini project develops an explainable machine-learning system for California house-value prediction. A Random Forest Regressor is trained on the California Housing dataset. Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R². Feature Importance, SHAP and Partial Dependence Plot (PDP) are used to make the model's behavior more understandable.

## 2. Problem Statement
A prediction system can produce a house-value estimate without showing which factors influenced it. This project aims to provide both prediction and understandable model explanations.

## 3. Objectives
- Predict median house values using machine learning.
- Evaluate model performance.
- Identify influential features.
- Provide global explanations using SHAP.
- Explain an individual prediction using a SHAP waterfall plot.
- Visualize the effect of median income using PDP.
- Discuss Responsible AI, bias and limitations.

## 4. Dataset
The California Housing dataset used in this project contains 20,640 observations. Input variables include longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income and ocean proximity. The target is median house value.

## 5. Methodology
1. Load the CSV dataset.
2. Handle missing numeric values using median imputation.
3. Handle the categorical ocean-proximity feature using most-frequent imputation and one-hot encoding.
4. Split the dataset into 80% training and 20% testing data.
5. Train a Random Forest Regressor.
6. Evaluate using MAE, RMSE and R².
7. Generate feature importance.
8. Generate SHAP summary and waterfall plots.
9. Generate a PDP for median income.

## 6. Algorithms
### Random Forest Regression
Random Forest uses an ensemble of decision trees to predict a continuous target.

### Feature Importance
The trained Random Forest provides a global ranking of feature influence.

### SHAP
SHAP assigns contribution values to features. The summary plot provides global information and the waterfall plot explains one prediction.

### Partial Dependence Plot
PDP shows how the model output changes as a selected feature changes.

## 7. Results
| Metric | Result |
|---|---:|
| MAE | 33,940.25 |
| RMSE | 51,238.93 |
| R² | 0.7996 |

The model was evaluated on 20% of the dataset after an 80:20 train-test split.

## 8. XAI Visualizations
Insert these generated figures:
1. `outputs/feature_importance.png`
2. `outputs/shap_summary.png`
3. `outputs/shap_waterfall.png`
4. `outputs/pdp_median_income.png`

The interpretation should be based on the generated plots. SHAP contribution values explain the model's behavior relative to its prediction baseline; they should not be interpreted as causal effects.

## 9. Responsible AI and Bias
The course material emphasizes transparency, fairness, privacy, security, reliability and accountability. Housing data can contain historical and geographic patterns. Therefore, a real deployment should check representativeness, geographic performance, error differences, data quality and model drift. Human review should be maintained for consequential decisions.

## 10. Limitations
- The model is intended as an academic demonstration.
- Random Forest explanations describe learned model behavior rather than causality.
- The dataset represents a historical housing context and may not reflect current market conditions.
- The model should not be used as an authoritative property valuation service without further validation.

## 11. Conclusion
The project demonstrates an end-to-end Explainable AI workflow. A Random Forest model provides house-value predictions, while Feature Importance, SHAP and PDP improve transparency. The project also considers responsible deployment issues such as bias, data representativeness and human oversight.

## 12. Future Enhancement
- Compare Random Forest with Gradient Boosting and Decision Tree models.
- Add LIME for local explanations.
- Build a Streamlit interface.
- Add geographic error analysis and fairness metrics.
- Add model monitoring.

## 13. References
1. Explainable AI and Responsible AI course material supplied for the course.
2. Explainable AI Algorithms and Code material supplied for the course.
3. Scikit-learn documentation.
4. SHAP documentation.
