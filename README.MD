# Foaming Capacity Prediction Using Machine Learning

This project aims to predict the **foaming capacity** of food ingredients based on their macronutrient composition using machine learning techniques. This could reduce the food product development time by moving away from trial-and-error approach towards a data-driven approach.

---

## Project Goals

- Predict foaming capacity (%) using ingredient macronutrient composition data
- Explore the relationship between macronutrients (protein, lipid, carbohydrate, fiber, ash) and foaming capability
- Compare performance of multiple regression models (Linear Regression, Elastic Net, Random Forest)

---

## Files

- [foaming_capacity_ml.ipynb](https://github.com/infoshanhe/ingredient-functionality-ml/blob/main/foaming_capacity_ml.ipynb): **This is the main file to look at.** I conducted the experiments in this file. All models, visualizations, analysis and conclusions can be found here.
- [cleaned_data.xlsx](https://github.com/infoshanhe/ingredient-functionality-ml/blob/main/cleaned_data.xlsx): Composition and functional data extracted from literature values from peer-reviewed publications
- [foaming_capacity_ml_utils.py](https://github.com/infoshanhe/ingredient-functionality-ml/blob/main/foaming_capacity_ml_utils.py): Functions used in the notebook

---

## Methods

- **Preprocessing**:
  - Missing value imputation
  - Standardization
  - Feature Engineering: Interaction effects
- **Modeling**:
  - Baseline: Linear Regression
  - Regularization: Elastic Net
  - Nonlinear: Random Forest
- **Validation**: Cross-validation with R²
- **Interpretation**:
  - Feature importance
  - Residuals and outlier analysis

---

## Results

- Best model: **linear regression with feature interaction** with mean R² ≈ 0.77
- Key predictors: Sample concentration, protein content
- Visualized model coefficients, correlations, and prediction residuals

---

## Key Learnings

- Functional properties like foaming capacity might be reasonably estimated from macronutrient profiles
- Public datasets are often sparse and noisy, requiring careful preprocessing and domain knowledge
- Data science tools can accelerate early-stage screening in ingredient research

---

