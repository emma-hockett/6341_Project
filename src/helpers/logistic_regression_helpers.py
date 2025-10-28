import pandas as pd
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

def scale_dataset(df: pd.DataFrame) -> pd.DataFrame:
    project_root = get_project_root()

    numeric_cols = ['loan_amount', 'income', 'combined_loan_to_value_ratio', 'loan_term', 'intro_rate_period',
                    'prepayment_penalty_term', 'property_value']

    scaler = joblib.load(os.path.join(project_root, 'models', 'scaler.pkl'))
    X_train_scaled = scaler.transform(df[numeric_cols])
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=df.index)

    return X_train_scaled_df



def transform_data_with_pca(df: pd.DataFrame) -> pd.DataFrame:
    ipca = joblib.load(os.path.join(get_project_root, 'models', 'ipca.pkl'))
    df_pca = ipca.transform(df)[:, :22]

    return df_pca


def get_project_root():
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    return project_root



def create_base_estimator():
    return LogisticRegression(
        solver="saga",  # Helps with larger datasets and elasticnet
        penalty="elasticnet",
        max_iter=5000,
        tol=1e-3,  # Loosening convergence criteria slightly to save compute/time
        n_jobs=-1,  # No limit on parallel threads
        random_state=42
    )



def create_param_grid() -> dict:
    param_grid = {
        "C": loguniform(1e-4, 1e2),
        "l1_ratio": uniform(0.0, 1.0),
        "class_weight": [None, "balanced"]
    }

    return param_grid



def create_cv_search():
    base_estimator = create_base_estimator()
    param_grid = create_param_grid()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    return RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=30,
        scoring="f1",
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=42
    )