# src/helpers/model_helpers.py
import src.utils.file_utils as fu
import pandas as pd

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