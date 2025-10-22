# src/helpers/eda_helpers.py
import pandas as pd
from scipy.stats import skew, kurtosis
import src.utils.schema_utils as su
import src.utils.file_utils as fu
import matplotlib.pyplot as plt


def get_numeric_features(df: pd.DataFrame, cfg_schema:object) -> pd.Series:
    # We left activity_year in for now, even though it's just one year.  Added this condition to avoid throwing warnings.
    return [
        c for c in su.get_columns_by_attribute(cfg_schema, "type", "numeric")
        if df[c].nunique(dropna=True) > 1
    ]


def get_numeric_columns_requiring_review(df: pd.DataFrame, numeric_cols:list[str]) -> pd.Series:
    cfg_eda = fu.load_config("eda")
    skew_threshold = cfg_eda["eda"]["skew_threshold"]
    kurtosis_threshold = cfg_eda["eda"]["kurtosis_threshold"]
    outlier_threshold = cfg_eda["eda"]["outlier_threshold"]

    return pd.DataFrame({
        "skew": identify_columns_with_skew(df, numeric_cols, skew_threshold),
        "kurtosis": identify_columns_with_kurtosis(df, numeric_cols, kurtosis_threshold),
        "outlier_pct": identify_columns_with_outliers(df, numeric_cols, outlier_threshold)
    })


def identify_columns_with_skew(df: pd.DataFrame, numeric_cols: list[str], skew_threshold: float) -> pd.Series:
    cols_skew = df[numeric_cols].apply(skew, nan_policy='omit')
    return cols_skew[cols_skew.abs() > skew_threshold]


def identify_columns_with_kurtosis(df: pd.DataFrame, numeric_cols: list[str], kurtosis_threshold: float) -> pd.Series:
    cols_kurtosis = df[numeric_cols].apply(kurtosis, nan_policy='omit')
    return cols_kurtosis[cols_kurtosis.abs() > kurtosis_threshold]


def identify_columns_with_outliers(df: pd.DataFrame, numeric_cols: list[str], outlier_threshold: float) -> pd.Series:
    outlier_info = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_pct = ((df[col] < lower) | (df[col] > upper)).mean()
        if outlier_pct > outlier_threshold:
            outlier_info[col] = outlier_pct

    return pd.Series(outlier_info, name='outlier_pct')


def plot_numeric_features(df: pd.DataFrame, numeric_cols_to_review: pd.DataFrame):
    for col, row in numeric_cols_to_review.iterrows():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        # We want to display a box plot - histogram - metrics across one row per feature
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.0, 2.2, 1.2], wspace=0.4)

        ax_box = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_txt = fig.add_subplot(gs[0, 2])

        # Boxplot
        ax_box.boxplot(s.values, vert=True, whis=1.5, showfliers=True)
        ax_box.set_xticks([])
        ax_box.set_title("Boxplot", fontsize=10)

        # Histogram
        ax_hist.hist(s.values, bins=40, density=True, alpha=0.8)
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Histogram", fontsize=10)

        # Text panel
        ax_txt.axis("off")
        ax_txt.text(0.0, 0.9, f"{col}", fontsize=12, fontweight="bold", transform=ax_txt.transAxes)
        ax_txt.text(0.0, 0.7, f"skew:       {row['skew']:.3f}", fontsize=10, transform=ax_txt.transAxes)
        ax_txt.text(0.0, 0.55, f"kurtosis:   {row['kurtosis']:.3f}", fontsize=10, transform=ax_txt.transAxes)
        ax_txt.text(0.0, 0.4, f"outlier %:  {row['outlier_pct']:.1%}", fontsize=10, transform=ax_txt.transAxes)
        ax_txt.text(0.0, 0.15, "visualized in [1st–99th pct]", fontsize=8, color="gray", transform=ax_txt.transAxes)

        # Adjust layout
        plt.subplots_adjust(left=0.08, right=0.95, wspace=0.4)
        plt.show()