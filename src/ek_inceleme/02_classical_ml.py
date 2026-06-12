"""
WESAD - Adım 6: Klasik ML benchmark'ı.

3 model × Leave-One-Subject-Out CV (15 fold):
  - Logistic Regression  (en basit, "öğrenmiş mi?" testi)
  - Random Forest        (genelde WESAD'da güçlü)
  - SVM (RBF kernel)     (yüksek-boyutlu non-lineer)

Subject-Independent: her seferinde 14 denek train, 1 denek test.
Bu yüzden raporladığımız accuracy = "yeni bir kişide nasıl çalışır?" sorusunun cevabı.

Çıktılar:
  outputs/cv_results.csv               - her fold × model satırı
  outputs/classification_reports.txt   - per-class precision/recall/F1
  figures/e02_confusion_matrices.png       - 3 model yan yana
  figures/e02_per_subject_accuracy.png     - hangi denek zor?
  figures/e02_feature_importance.png       - RF'in en çok kullandığı özellikler
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

CLASS_NAMES = {1: "Baseline", 2: "Stress", 3: "Amusement"}
RANDOM_STATE = 42


def make_models() -> dict:
    """Üç model — hepsi class_weight='balanced' ile imbalance'a karşı."""
    return {
        "LogReg": lambda: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "SVM-RBF": lambda: SVC(
            kernel="rbf", class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           feature_names: list[str]) -> tuple[pd.DataFrame, dict, dict, np.ndarray]:
    """LOSO CV. Returns per-fold sonuçlar, model→toplam_predictions, RF importance, true_labels."""
    logo = LeaveOneGroupOut()
    models = make_models()

    fold_results: list[dict] = []
    all_true_per_fold: list[np.ndarray] = []
    all_pred_per_model: dict[str, list[np.ndarray]] = {m: [] for m in models}
    rf_importances: list[np.ndarray] = []

    n_folds = logo.get_n_splits(X, y, groups)
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        test_subject = int(groups[test_idx][0])
        y_test = y[test_idx]
        all_true_per_fold.append(y_test)

        for name, factory in models.items():
            pipe = make_pipeline(StandardScaler(), factory())
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            all_pred_per_model[name].append(y_pred)

            fold_results.append({
                "fold": fold,
                "test_subject": test_subject,
                "model": name,
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            })

            # Sadece RF'den importance topla
            if name == "RandomForest":
                rf = pipe.named_steps["randomforestclassifier"]
                rf_importances.append(rf.feature_importances_)

        print(f"  Fold {fold:2d}/{n_folds} (test=S{test_subject}): "
              + " | ".join(
                  f"{m}={fold_results[-len(models) + i]['accuracy']:.3f}"
                  for i, m in enumerate(models)
              ))

    df = pd.DataFrame(fold_results)
    all_true = np.concatenate(all_true_per_fold)
    all_pred = {m: np.concatenate(preds) for m, preds in all_pred_per_model.items()}
    rf_imp_mean = np.mean(rf_importances, axis=0)
    return df, all_pred, dict(zip(feature_names, rf_imp_mean)), all_true


def plot_confusion_matrices(all_true: np.ndarray, all_pred: dict, out_path: Path) -> None:
    labels = [1, 2, 3]
    names = [CLASS_NAMES[l] for l in labels]

    n_models = len(all_pred)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, y_pred) in zip(axes, all_pred.items()):
        cm = confusion_matrix(all_true, y_pred, labels=labels)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                        ha="center", va="center", color=color, fontsize=10)

        acc = accuracy_score(all_true, y_pred)
        f1 = f1_score(all_true, y_pred, average="macro")
        ax.set_title(f"{model_name}\nacc={acc:.3f}  macro-F1={f1:.3f}", fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("Confusion Matrix — LOSO CV birleşik", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_subject_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="test_subject", columns="model", values="accuracy").sort_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(pivot.index))
    width = 0.25
    colors = {"LogReg": "#8E8E8E", "RandomForest": "#4C72B0", "SVM-RBF": "#C44E52"}

    for i, model in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * width, pivot[model].values, width,
               label=model, color=colors.get(model, None))

    ax.axhline(0.55, color="orange", linestyle="--", alpha=0.7, label="Majority baseline (~%55)")
    ax.axhline(0.33, color="red", linestyle=":", alpha=0.6, label="Random guess (%33)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in pivot.index], rotation=0)
    ax.set_ylabel("Accuracy (test edilen denekte)")
    ax.set_xlabel("Denek (LOSO test fold)")
    ax.set_title("LOSO Accuracy — Denek Başına")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_dict: dict, out_path: Path) -> None:
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*items)
    y_pos = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y_pos, values, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Ortalama Importance (15 fold)")
    ax.set_title("Random Forest — Özellik Önemi")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(INPUT_PATH)

    feature_cols = [c for c in df.columns if c not in ("label", "subject", "start_sec")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int8)
    groups = df["subject"].values

    print(f"  X shape: {X.shape}  (3040, 18 özellik)")
    print(f"  Sınıf dağılımı: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Denek sayısı: {len(np.unique(groups))}")

    # Baseline'ları yazdır
    majority_class = pd.Series(y).mode()[0]
    majority_acc = (y == majority_class).mean()
    print(f"\nBaseline'lar (modelin geçmesi gerekenler):")
    print(f"  Random guess (1/3):      0.333")
    print(f"  Majority class (Always Baseline): {majority_acc:.3f}")

    print(f"\nLOSO CV başlıyor (15 fold × 3 model):")
    cv_df, all_pred, rf_imp, all_true = run_cv(X, y, groups, feature_cols)

    # Kaydet
    cv_df.to_csv(OUTPUTS_DIR / "cv_results.csv", index=False)

    # Genel özet
    print("\n=== Genel sonuçlar (15 fold ortalaması) ===")
    summary = cv_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_weighted_mean=("f1_weighted", "mean"),
    ).round(3).sort_values("accuracy_mean", ascending=False)
    print(summary.to_string())

    # Classification reports
    report_text = []
    for model_name, y_pred in all_pred.items():
        report_text.append(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
        report_text.append(classification_report(
            all_true, y_pred,
            labels=[1, 2, 3],
            target_names=["Baseline", "Stress", "Amusement"],
            digits=3,
        ))
    report_path = OUTPUTS_DIR / "classification_reports.txt"
    report_path.write_text("\n".join(report_text), encoding="utf-8")
    print(f"\n[OK] Classification reports: {report_path.relative_to(PROJECT_ROOT)}")

    # Görseller
    plot_confusion_matrices(all_true, all_pred, FIGURES_DIR / "e02_confusion_matrices.png")
    plot_per_subject_accuracy(cv_df, FIGURES_DIR / "e02_per_subject_accuracy.png")
    plot_feature_importance(rf_imp, FIGURES_DIR / "e02_feature_importance.png")

    print("\n[OK] Görseller: e02_confusion_matrices.png, e02_per_subject_accuracy.png, e02_feature_importance.png")
    print(f"[OK] CV detayı: {(OUTPUTS_DIR / 'cv_results.csv').relative_to(PROJECT_ROOT)}")

    # En önemli 5 özellik
    print("\n=== Random Forest'in en önemli 5 özelliği ===")
    top5 = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, imp in top5:
        print(f"  {name:14s} : {imp:.3f}")


if __name__ == "__main__":
    main()
