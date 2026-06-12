"""
WESAD - Adım 7: Klasik ML iyileştirmeleri.

3 bölüm:
  1. S13 anomalisi: feature z-score analizi (tüm modeller orada başarısızdı, neden?)
  2. HistGradientBoosting eklenir (XGBoost-benzeri, sklearn-native)
       + RF/SVM hiperparametreleri biraz daha sıkı
  3. Binary task (Stress vs Non-stress) — daha kolay problem, daha net hikaye

Çıktılar:
  outputs/cv_results_extended.csv   - 3class + binary, 4 model
  figures/e03_s13_vs_others.png         - S13 sapması görseli
  figures/e03_model_comparison.png      - 3class vs binary bar chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "features.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42
MODEL_COLORS = {"LogReg": "#8E8E8E", "RandomForest": "#4C72B0",
                "HistGradBoost": "#55A868", "SVM-RBF": "#C44E52"}


# =====================================================================
# Bölüm 1: S13 anomalisi
# =====================================================================

def investigate_s13(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """S13'ün her özellikteki ortalamasını diğerlerinin dağılımına göre z-score'la."""
    s13 = df[df["subject"] == 13][feature_cols]
    others = df[df["subject"] != 13][feature_cols]
    mu = others.mean()
    sigma = others.std()
    z = ((s13.mean() - mu) / sigma)
    return {
        "z_scores": z.to_dict(),
        "s13_mean": s13.mean().to_dict(),
        "others_mean": mu.to_dict(),
        "others_std": sigma.to_dict(),
        "top_deviations": z.abs().sort_values(ascending=False).to_dict(),
    }


def plot_s13_vs_others(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    s13_mask = df["subject"] == 13
    others = df[~s13_mask][feature_cols]
    s13 = df[s13_mask][feature_cols]
    z = ((s13.mean() - others.mean()) / others.std()).abs()
    top6 = z.sort_values(ascending=False).head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feat in zip(axes.flat, top6):
        data = [df[~s13_mask][feat].values, df[s13_mask][feat].values]
        bp = ax.boxplot(data, tick_labels=["Diğer 14 denek", "S13"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#C44E52")
        ax.set_title(f"{feat}  (|z|={z[feat]:.2f})")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("S13'ün En Sapan 6 Özelliği — Box Plot Karşılaştırma", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Bölüm 2 & 3: Modeller ve CV
# =====================================================================

def make_models() -> dict:
    """4 model. RF ve HGB için daha sıkı hiperparametreler."""
    return {
        "LogReg": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0, random_state=RANDOM_STATE
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradBoost": lambda: HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.08,
            l2_regularization=0.1, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM-RBF": lambda: SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task_name: str
           ) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """LOSO CV — tüm modeller, tüm fold'lar."""
    logo = LeaveOneGroupOut()
    models = make_models()
    fold_results: list[dict] = []
    all_true_per_fold: list[np.ndarray] = []
    all_pred_per_model: dict[str, list[np.ndarray]] = {m: [] for m in models}

    for fold, (tr, te) in enumerate(logo.split(X, y, groups), start=1):
        test_subj = int(groups[te][0])
        y_test = y[te]
        all_true_per_fold.append(y_test)
        for name, factory in models.items():
            pipe = make_pipeline(StandardScaler(), factory())
            pipe.fit(X[tr], y[tr])
            y_pred = pipe.predict(X[te])
            all_pred_per_model[name].append(y_pred)
            fold_results.append({
                "task": task_name, "fold": fold, "test_subject": test_subj,
                "model": name, "n_test": len(te),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            })

    return (pd.DataFrame(fold_results),
            {m: np.concatenate(p) for m, p in all_pred_per_model.items()},
            np.concatenate(all_true_per_fold))


def plot_model_comparison(s3: pd.DataFrame, sbin: pd.DataFrame,
                           bin_baseline: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, summary, title, baseline in [
        (axes[0], s3, "3-sınıf (Baseline / Stress / Amusement)", 0.55),
        (axes[1], sbin, "Binary (Stress vs Non-stress)", bin_baseline),
    ]:
        models = summary.index.tolist()
        x = np.arange(len(models))
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        ax.bar(x, summary["acc"].values, yerr=summary["acc_std"].values,
               capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10)
        ax.axhline(baseline, color="orange", linestyle="--",
                   label=f"Majority baseline ({baseline:.2f})")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        for xi, val in zip(x, summary["acc"].values):
            ax.text(xi, val + 0.015, f"{val:.3f}", ha="center",
                    fontsize=9, fontweight="bold")
    fig.suptitle("Model Karşılaştırması — Genişletilmiş", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    feature_cols = [c for c in df.columns if c not in ("label", "subject", "start_sec")]
    print(f"  {len(df)} satır × {len(feature_cols)} özellik\n")

    # ----- Bölüm 1: S13 -----
    print("=" * 60)
    print("Bölüm 1: S13 anomalisi (z-score)")
    print("=" * 60)
    diag = investigate_s13(df, feature_cols)

    print("\nS13'ün en sapan 6 özelliği (|z| büyükten küçüğe):")
    for feat in list(diag["top_deviations"].keys())[:6]:
        z = diag["z_scores"][feat]
        s13_val = diag["s13_mean"][feat]
        oth_val = diag["others_mean"][feat]
        oth_std = diag["others_std"][feat]
        arrow = "↑" if z > 0 else "↓"
        print(f"  {feat:14s}: z={z:+.2f} {arrow}  "
              f"(S13={s13_val:.2f}, diğerleri={oth_val:.2f}±{oth_std:.2f})")

    plot_s13_vs_others(df, feature_cols, FIGURES_DIR / "e03_s13_vs_others.png")
    print("\n[OK] Görsel: figures/e03_s13_vs_others.png")

    # ----- Bölüm 2: 3-sınıf -----
    print("\n" + "=" * 60)
    print("Bölüm 2: 3-sınıf — 4 model (LogReg, RF, HistGradBoost, SVM)")
    print("=" * 60)

    X = df[feature_cols].values.astype(np.float32)
    y_3 = df["label"].values.astype(np.int8)
    groups = df["subject"].values

    print("LOSO CV başlıyor...")
    cv3, pred3, true3 = run_cv(X, y_3, groups, "3class")
    s3 = cv3.groupby("model").agg(
        acc=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1=("f1_macro", "mean"), f1_w=("f1_weighted", "mean"),
    ).round(3).sort_values("acc", ascending=False)
    print("\n3-sınıf özet:")
    print(s3.to_string())

    # ----- Bölüm 3: Binary -----
    print("\n" + "=" * 60)
    print("Bölüm 3: Binary — Stress (1) vs Non-stress (0)")
    print("=" * 60)

    y_bin = (y_3 == 2).astype(np.int8)
    n_stress = int(y_bin.sum())
    n_nonstress = int(len(y_bin) - n_stress)
    bin_baseline = n_nonstress / len(y_bin)
    print(f"Stress={n_stress}, Non-stress={n_nonstress}, "
          f"majority baseline={bin_baseline:.3f}")

    print("LOSO CV başlıyor...")
    cv_bin, pred_bin, true_bin = run_cv(X, y_bin, groups, "binary")
    sbin = cv_bin.groupby("model").agg(
        acc=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1=("f1_macro", "mean"), f1_w=("f1_weighted", "mean"),
    ).round(3).sort_values("acc", ascending=False)
    print("\nBinary özet:")
    print(sbin.to_string())

    # ----- Bölüm 4: Kaydet & görselleştir -----
    pd.concat([cv3, cv_bin]).to_csv(OUTPUTS_DIR / "cv_results_extended.csv", index=False)
    print(f"\n[OK] CSV: outputs/cv_results_extended.csv")

    plot_model_comparison(s3, sbin, bin_baseline,
                           FIGURES_DIR / "e03_model_comparison.png")
    print("[OK] Görsel: figures/e03_model_comparison.png")

    # En iyi modelin binary classification report'u
    best_bin_model = sbin.index[0]
    print(f"\n=== En iyi binary modelin detayı ({best_bin_model}) ===")
    print(classification_report(true_bin, pred_bin[best_bin_model],
                                 target_names=["Non-stress", "Stress"], digits=3))


if __name__ == "__main__":
    main()
