import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "Book1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_charts")
MODEL_COL  = "models"             # name of the column holding model names

# Distinct colors per model (extend this list if you add more models)
MODEL_COLORS = {
    "TransMobile": "Blue",   # blue
    "SwinTrans"  : "Green",   # green
    "IM RCNN"    : "Pink",   # orange
}
DEFAULT_COLOR_CYCLE = ["Blue", "green", "Orange", "Brown", "Yellow"]

# Metrics where LOWER is better (flagged red + annotated, like Error Rate)
LOWER_IS_BETTER = {"error-rate", "error_rate", "errorrate"}


def load_data(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Update CSV_PATH at the top of this script."
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if MODEL_COL not in df.columns:
        raise ValueError(
            f"Expected a '{MODEL_COL}' column, found: {list(df.columns)}"
        )
    return df


def get_model_colors(model_names):
    colors = []
    for i, name in enumerate(model_names):
        colors.append(MODEL_COLORS.get(name, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)]))
    return colors


def plot_single_metric(df, metric_col, output_dir=OUTPUT_DIR):
    models = df[MODEL_COL].tolist()
    values = df[metric_col].astype(float).tolist()
    colors = get_model_colors(models)

    is_lower_better = metric_col.strip().lower() in LOWER_IS_BETTER
    if is_lower_better:
        colors = ["red"] * len(models)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(values) * 0.015,
                 f"{val:.2f}%",
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    title = metric_col + (" (lower = better)" if is_lower_better else "")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(fontsize=11, rotation=10)
    plt.tight_layout()

    safe_name = metric_col.replace("/", "_").replace(" ", "_").replace("@", "at")
    out_path  = os.path.join(output_dir, f"comparison_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def plot_all_metrics(csv_path=CSV_PATH, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(csv_path)

    metric_cols = [c for c in df.columns if c != MODEL_COL]
    print(f"  Loaded {len(df)} models, {len(metric_cols)} metrics: {metric_cols}\n")

    saved_paths = []
    for metric_col in metric_cols:
        saved_paths.append(plot_single_metric(df, metric_col, output_dir))

    print(f"\n  Done. {len(saved_paths)} comparison charts saved to: {os.path.abspath(output_dir)}")
    return saved_paths


if __name__ == "__main__":
    plot_all_metrics()
