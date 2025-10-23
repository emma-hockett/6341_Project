# src/helpers/eda_helpers.py
import pandas as pd
from scipy.stats import skew, kurtosis
import src.utils.schema_utils as su
import src.utils.file_utils as fu
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import pointbiserialr


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


def get_categorical_columns_requiring_review(df: pd.DataFrame, category_cols: list[str], category_min_pct_threshold: float) -> pd.DataFrame:
    low_frequency_features = []
    for col in category_cols:
        counts = df[col].value_counts(dropna=False)
        n_unique = counts.size
        min_freq = counts.min()

        # We calculate the minimum percentage of observed values
        valid = df[col].notna().sum()
        min_pct = min_freq / valid if valid > 0 else 0

        if min_pct < 0.001:
            low_frequency_features.append({
                "feature": col,
                "n_unique": n_unique,
                "min_freq": min_freq,
                "min_pct": min_pct
            })

    return pd.DataFrame(low_frequency_features)



def identify_perfect_category_classes(df, category_cols, target):
    MIN_N = 50
    rows = []
    for col in category_cols:
        g = df.groupby(col, dropna=False)[target].agg(rate='mean', n='count')
        if len(g) < 2:
            continue
        mask = ((g['rate'] == 0) | (g['rate'] == 1)) & (g['n'] >= MIN_N)
        if mask.any():
            for cls, r in g[mask].iterrows():
                rows.append({'feature': col, 'class_label': cls, 'n': int(r['n']), 'denial_rate': r['rate']})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['feature','denial_rate','n'], ascending=[True, True, False]).reset_index(drop=True)
    return out


def identify_categories_with_large_denial_ranges(df, category_cols, target):
    RANGE_T = 0.15
    rows = []
    for col in category_cols:
        g = df.groupby(col, dropna=False)[target].agg(rate='mean', n='count')
        if len(g) < 2:
            continue
        rmin, rmax = g['rate'].min(), g['rate'].max()
        frange = rmax - rmin
        if frange > RANGE_T:
            rows.append({
                'feature': col,
                'n_classes': len(g),
                'min_rate': rmin,
                'max_rate': rmax,
                'feature_range': frange
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('feature_range', ascending=False).reset_index(drop=True)
    return out


def identify_highly_correlated_numeric_features(df: pd.DataFrame, numeric_cols: list[str]):
    corr_matrix = df[numeric_cols].corr(method='pearson').abs()

    high_corr = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'corr'})
    )
    return high_corr[high_corr['corr'] > 0.8].sort_values('corr', ascending=False)


def identify_feature_target_correlations(df: pd.DataFrame, numeric_cols: list[str], target: str):
    results_num = []
    for col in numeric_cols:
        # Drop NaNs for valid pairs
        valid = df[[col, target]].dropna()
        if valid[col].nunique() > 1:
            r, p = pointbiserialr(valid[col], valid[target])
            results_num.append({'feature': col, 'corr': r, 'p_value': p})

    return pd.DataFrame(results_num).sort_values('corr', key=abs, ascending=False)


def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def identify_feature_target_correlations(df: pd.DataFrame, category_cols: list[str], target: str):
    results_cat = []
    for col in category_cols:
        if df[col].nunique() > 1:
            v = cramers_v(df[col].astype(str), df[target])
            results_cat.append({'feature': col, 'cramers_v': v})

    return pd.DataFrame(results_cat).sort_values('cramers_v', ascending=False)