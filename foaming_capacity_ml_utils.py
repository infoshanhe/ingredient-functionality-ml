import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def plot_rf_importances(rf_model: RandomForestClassifier | RandomForestRegressor):
    importances = rf_model.feature_importances_
    importance_series = pd.Series(importances, index=rf_model.feature_names_in_).sort_values(ascending=True)

    plt.figure(figsize=(8, 5), dpi=600)
    importance_series.plot(kind='barh')
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def print_outliers(outlier_mask, y_pred_cv,X, y, residuals):
    print('Printing info on outliers:\n')
    print("Number of outliers:", outlier_mask.sum(), end='\n\n')
    outliers = X[outlier_mask]
    print(f"Is plant ingredient? 0 = False, 1 = True\n{outliers['Plant ingredient?']}\n")
    print(f"True values:\n{y[outlier_mask]}\n")
    print(f"Predicted values:\n{pd.Series(y_pred_cv, index=X.index)[outlier_mask]}\n")
    print(f"Residuals:\n{residuals[outlier_mask]}")

def print_per_fold_coefficients(cv_results, feature_names):
    print("Per-fold coefficients:")
    for i, model in enumerate(cv_results['estimator']):
        coefs = model.named_steps['model'].coef_
        print(f"\nFold {i+1} R²: {cv_results['test_score'][i]:.2f}")
        for name, coef in zip(feature_names, coefs):
            print(f"  {name}: {coef:.2f}")

def print_and_plot_coef_values(cv_results ,feature_names):
    coefs_per_fold = np.array([
        model.named_steps['model'].coef_
        for model in cv_results['estimator']
    ])

    # Standard error: std deviation across folds divided by sqrt(n_folds)
    n_folds = coefs_per_fold.shape[0]
    coef_se = coefs_per_fold.std(axis=0, ddof=1) / np.sqrt(n_folds)
    print("Mean coefficient values:", coefs_per_fold.mean(axis=0))
    print("Standard error of coefficients:", coef_se)


    plt.figure(figsize=(10, 5),dpi=600)

    for i in range(coefs_per_fold.shape[1]):  # loop over features
        plt.plot(coefs_per_fold[:, i], marker='o', label=feature_names[i])

    plt.title("Coefficient values across folds")
    plt.xlabel("Fold number")
    plt.xticks(ticks=range(n_folds), labels=[f"{i+1}" for i in range(n_folds)])
    plt.ylabel("Coefficient value")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_feature_correlation_matrix(X):
    corr = X.corr()
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.xticks([])
    plt.title("Feature Correlation Matrix")
    plt.show()