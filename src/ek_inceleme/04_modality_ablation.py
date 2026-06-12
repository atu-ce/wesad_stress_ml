"""
WESAD - Adım 8: Modalite (sensör grubu) ablation çalışması.

Soru: "Fabrika işçisine sadece akıllı bileklik versek yeter mi?"

6 sensör kombinasyonu × 2 model (RF, LogReg) × 2 task (3-sınıf, binary) × 15 fold

Modalite grupları:
  - All sensors           : 18 özellik (baseline — Adım 7'deki tam çözüm)
  - Chest only            : 13 özellik (göğüs bandı tek başına)
  - Wrist only            :  5 özellik (akıllı saat tek başına) ⭐ ana sorum
  - Heart only (HR/HRV)   :  9 özellik (sadece kalp kaynaklı)
  - EDA only              :  7 özellik (sadece deri iletkenliği)
  - Resp only             :  2 özellik (sadece solunum)

Çıktılar:
  outputs/ablation_results.csv      - tüm kombinasyonlar
  figures/e04_modality_comparison.png   - sensör grubu × accuracy bar chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "features.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42

ALL_FEATURES = [
    "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf", "hrv_hf", "hrv_lfhf",
    "scl_mean", "scl_std", "scr_count", "scr_mean_amp",
    "resp_rate", "resp_rate_std",
    "bvp_hr_mean", "bvp_hrv_sdnn",
    "scl_w_mean", "scl_w_std", "scl_w_slope",
]

MODALITY_GROUPS = {
    "All sensors": ALL_FEATURES,
    "Chest only": [
        "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf", "hrv_hf", "hrv_lfhf",
        "scl_mean", "scl_std", "scr_count", "scr_mean_amp",
        "resp_rate", "resp_rate_std",
    ],
    "Wrist only": [
        "bvp_hr_mean", "bvp_hrv_sdnn",
        "scl_w_mean", "scl_w_std", "scl_w_slope",
    ],
    "Heart only (HR/HRV)": [
        "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
        "hrv_lf", "hrv_hf", "hrv_lfhf",
        "bvp_hr_mean", "bvp_hrv_sdnn",
    ],
    "EDA only": [
        "scl_mean", "scl_std", "scr_count", "scr_mean_amp",
        "scl_w_mean", "scl_w_std", "scl_w_slope",
    ],
    "Resp only": ["resp_rate", "resp_rate_std"],
}


def make_models() -> dict:
    return {
        "LogReg": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def run_one_combo(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   modality_name: str, task_name: str) -> list[dict]:
    """Bir (modality, task) kombinasyonu için tüm modeller × tüm fold'lar."""
    logo = LeaveOneGroupOut()
    models = make_models()
    rows: list[dict] = []
    for fold, (tr, te) in enumerate(logo.split(X, y, groups), start=1):
        test_subj = int(groups[te][0])
        y_test = y[te]
        for model_name, factory in models.items():
            pipe = make_pipeline(StandardScaler(), factory())
            pipe.fit(X[tr], y[tr])
            y_pred = pipe.predict(X[te])
            rows.append({
                "modality": modality_name,
                "n_features": X.shape[1],
                "task": task_name,
                "model": model_name,
                "fold": fold,
                "test_subject": test_subj,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
            })
    return rows


def plot_modality_comparison(df: pd.DataFrame, baselines: dict, out_path: Path) -> None:
    """Modalite × task × model bar chart."""
    summary = df.groupby(["modality", "task", "model"]).agg(
        acc=("accuracy", "mean"), acc_std=("accuracy", "std"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    modality_order = list(MODALITY_GROUPS.keys())
    n_feat_by_modality = {k: len(v) for k, v in MODALITY_GROUPS.items()}
    x = np.arange(len(modality_order))
    width = 0.36

    for ax, task, title in [(axes[0], "3class", "3-sınıf (Baseline / Stress / Amusement)"),
                             (axes[1], "binary", "Binary (Stress vs Non-stress)")]:
        sub = summary[summary["task"] == task]
        for i, model in enumerate(["LogReg", "RandomForest"]):
            mdl = sub[sub["model"] == model].set_index("modality").reindex(modality_order)
            color = "#8E8E8E" if model == "LogReg" else "#4C72B0"
            ax.bar(x + (i - 0.5) * width, mdl["acc"].values,
                   width, yerr=mdl["acc_std"].values, capsize=3,
                   color=color, label=model, edgecolor="black", linewidth=0.4)

        ax.axhline(baselines[task], color="orange", linestyle="--",
                   label=f"Majority baseline ({baselines[task]:.2f})")
        # x labels: modality + n_features
        labels = [f"{m}\n({n_feat_by_modality[m]} özellik)" for m in modality_order]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy" if task == "3class" else "")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        # RF değerlerini bar üstüne yaz
        rf_data = sub[sub["model"] == "RandomForest"].set_index("modality").reindex(modality_order)
        for xi, val in zip(x, rf_data["acc"].values):
            ax.text(xi + 0.5 * width, val + 0.015, f"{val:.3f}",
                    ha="center", fontsize=8, fontweight="bold", color="#4C72B0")

    fig.suptitle("Modalite Ablation — Hangi Sensör Kombinasyonu Yeter?", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    y_3 = df["label"].values.astype(np.int8)
    y_bin = (y_3 == 2).astype(np.int8)
    groups = df["subject"].values

    bin_baseline = float((y_bin == 0).mean())
    print(f"  {len(df)} pencere, {len(np.unique(groups))} denek")
    print(f"  3-sınıf majority baseline: 0.550")
    print(f"  Binary  majority baseline: {bin_baseline:.3f}\n")

    print(f"6 modalite × 2 task × 2 model × 15 fold = {6 * 2 * 2 * 15} fit\n")

    all_rows: list[dict] = []
    for modality_name, feature_list in MODALITY_GROUPS.items():
        X = df[feature_list].values.astype(np.float32)
        for task_name, y in [("3class", y_3), ("binary", y_bin)]:
            print(f"  {modality_name:25s}  [{task_name:7s}]  "
                  f"({len(feature_list)} özellik)...", end=" ", flush=True)
            rows = run_one_combo(X, y, groups, modality_name, task_name)
            all_rows.extend(rows)
            # Anlık özet
            rf_acc = np.mean([r["accuracy"] for r in rows if r["model"] == "RandomForest"])
            print(f"RF acc={rf_acc:.3f}")

    ablation_df = pd.DataFrame(all_rows)
    ablation_df.to_csv(OUTPUTS_DIR / "ablation_results.csv", index=False)

    # Özet tablo
    print("\n" + "=" * 70)
    print("Özet Tablo (RandomForest, 15-fold ortalama)")
    print("=" * 70)
    pivot = ablation_df[ablation_df["model"] == "RandomForest"].pivot_table(
        index="modality", columns="task", values="accuracy", aggfunc="mean"
    ).round(3).reindex(MODALITY_GROUPS.keys())
    pivot.columns = ["3-sınıf", "Binary"]
    pivot.insert(0, "Özellik sayısı",
                 [len(MODALITY_GROUPS[m]) for m in pivot.index])
    print(pivot.to_string())

    # Wrist-only ile All sensors karşılaştırması
    wrist_bin = ablation_df[(ablation_df["modality"] == "Wrist only")
                            & (ablation_df["task"] == "binary")
                            & (ablation_df["model"] == "RandomForest")]["accuracy"].mean()
    all_bin = ablation_df[(ablation_df["modality"] == "All sensors")
                          & (ablation_df["task"] == "binary")
                          & (ablation_df["model"] == "RandomForest")]["accuracy"].mean()
    gap = (all_bin - wrist_bin) * 100
    print(f"\n💡 Wrist-only kayıp (binary): {wrist_bin:.3f} vs {all_bin:.3f} → -{gap:.1f} puan")

    plot_modality_comparison(ablation_df,
                              {"3class": 0.55, "binary": bin_baseline},
                              FIGURES_DIR / "e04_modality_comparison.png")
    print("\n[OK] CSV: outputs/ablation_results.csv")
    print("[OK] Görsel: figures/e04_modality_comparison.png")


if __name__ == "__main__":
    main()
