# Bias Detection & Interpretability


### Overview
**Category:** Bias Detection

**Modular Components:**
- Fairness Metrics Evaluator
- Model Explainer (SHAP/LIME)
- Feature Attribution Visualizer

### Use Cases
- Do public service recommendation models (e.g., bus stop placement, streetlight allocation) tend to ignore low-density or low-income areas?
- Why citizens' application for housing subsidies or public services was deprioritized?
- Do predictive maintenance systems for infrastructure (e.g., water leaks, power outages) prioritize certain zones? Is this optimization fair?
- In citizen feedback systems (e.g., 311 complaints), are certain types of reports more likely to trigger response recommendations than others? Why?

### Code Example: Credit Decision Bias Auditing
**Content:**
- Analyze credit data using LLM and interpretable ML
- Detect bias in approval logic (e.g., income, gender)
- Apply SHAP and counterfactual fairness methods

**Datasets:**
- German Dataset (credit data) from AIF Fairness 360
- COMPAS dataset (pre-prepared from https://www.kaggle.com/datasets/danofer/compass)

**Required Packages:** Fairlearn, SHAP, pandas, scikit-learn, transformers
