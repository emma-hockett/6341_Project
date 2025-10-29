import pandas as pd
import src.utils.file_utils as fu

def load_model_metrics_from_csv(model: str):
    model_path = fu.get_path(f"{model}_metrics_csv")
    df = pd.read_csv(model_path)
    df.columns = ["metric", "Score"]

    df_wide = df.set_index("metric").T
    df_wide["Model"] = model
    df_wide = df_wide.set_index("Model")

    return df_wide