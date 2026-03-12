import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

def load_results():
    path = os.path.join(RESULTS, "benchmark.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run benchmark/compare.py first.")
    return pd.read_csv(path)

def plot_dashboard(df):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("PTQ Pipeline — FP32 vs INT8 on Apple Silicon (CPU)", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)
    colors = ["#4C72B0", "#DD8452"]

    # Size
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(df["Model"], df["Size (MB)"], color=colors, width=0.4)
    ax1.set_title("Model Size (MB)")
    ax1.set_ylabel("MB")
    for i, v in enumerate(df["Size (MB)"]):
        ax1.text(i, v + 0.3, f"{v} MB", ha="center", fontsize=10)

    # Latency
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(df["Model"], df["Latency (ms)"], yerr=df["Latency Std"],
            color=colors, width=0.4, capsize=5)
    ax2.set_title("Inference Latency (ms)")
    ax2.set_ylabel("ms (lower = better)")
    for i, v in enumerate(df["Latency (ms)"]):
        ax2.text(i, v + 0.5, f"{v} ms", ha="center", fontsize=10)

    # Trade-off scatter
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(df["Size (MB)"], df["Latency (ms)"], c=colors, s=200, zorder=3)
    for _, row in df.iterrows():
        ax3.annotate(row["Model"], (row["Size (MB)"], row["Latency (ms)"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax3.set_title("Latency vs. Size Trade-off")
    ax3.set_xlabel("Model Size (MB)")
    ax3.set_ylabel("Latency (ms)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    out = os.path.join(RESULTS, "ptq_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Dashboard saved → {out}")
    plt.show()

if __name__ == "__main__":
    df = load_results()
    print("\n── Profiling Summary ──────────────────────")
    print(df[["Model","Size (MB)","Latency (ms)","Size Reduction","Speedup"]].to_string(index=False))
    plot_dashboard(df)
