# src/helpers/model_helpers.py
import src.utils.file_utils as fu
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

def load_model_dataset():
    modeling_dataset = fu.load_parquet("hmda_2024_model")
    train_output_path = fu.get_path("train_index")
    test_output_path = fu.get_path("test_index")
    train_idx = pd.read_csv(train_output_path)["index"]
    test_idx = pd.read_csv(test_output_path)["index"]

    # Subset the DataFrame
    train_df = modeling_dataset.loc[train_idx]
    test_df = modeling_dataset.loc[test_idx]

    # This is temporary to test functionality on a very small sample.  Remove before running final training.
    train_df = train_df.sample(frac=0.005, random_state=42)
    test_df = test_df.sample(frac=0.005, random_state=42)

    # Separate X and y
    target_col = "denied_flag"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test


def persist_model(model_selector, path_key: str):
    model_path = fu.get_path(path_key)
    joblib.dump(model_selector.best_estimator_, model_path)


def save_metrics_to_csv(results, key: str):
    csv_path = fu.get_path(key)
    results.to_csv(csv_path, index=True)


def save_viz(plot, key: str):
    path = fu.get_path(key)
    plot.savefig(path, dpi=300, bbox_inches="tight")


def output_cv_summary(model_selector):
    print("Best params:", model_selector.best_params_)
    print("Best CV F1:", model_selector.best_score_)


def calculate_test_metrics(model_selector, X_test, y_test):
    best_lr = model_selector.best_estimator_
    y_pred = best_lr.predict(X_test)
    y_prob = best_lr.predict_proba(X_test)[:, 1]

    metrics = {
        "F1": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
        "PR AUC": average_precision_score(y_test, y_prob)
    }

    results = pd.DataFrame(metrics, index=["Score"]).T

    # Return metrics
    return results, y_pred, y_prob


def draw_roc_curve(y_test, y_prob, output_path_key):
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    save_viz(plt, output_path_key)
    plt.show()


def draw_pr_curve(y_test, y_prob, output_path_key):
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve")
    save_viz(plt, output_path_key)
    plt.show()