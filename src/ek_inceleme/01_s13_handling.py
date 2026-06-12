"""
WESAD - Adım 9: S13 anomalisini ele almanın 4 yolunu karşılaştır.

S13'ün bilek EDA sensörü bozuk (z=+3.46). 4 strateji:
  1. As-is        : hiçbir şey yapma (mevcut pipeline — referans)
  2. Per-subject  : her deneği KENDİ içinde z-normalize et (anomaliyi nötrle)
  3. Winsorize    : her özelliği train'in %1-%99 aralığına kırp (uçları törpüle)
  4. Remove S13   : S13'ü tamamen çıkar (en basit, ama 14 denek kalır)

Her strateji × 2 task (3-sınıf, binary) × RandomForest × LOSO CV.

Adil kıyas için 3 metrik:
  - Tüm ort.     : sahip olduğu tüm fold'ların ortalaması
  - Diğer-14 ort.: S13 HARİÇ 14 deneğin ortalaması (4 yöntemde de karşılaştırılabilir)
  - S13 fold     : sadece S13 test edildiğindeki accuracy (yöntem S13'ü düzeltti mi?)

Çıktılar:
  outputs/s13_handling_results.csv
  figures/e01_s13_handling_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "features.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def per_subject_normalize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Her deneğin özelliklerini KENDİ ortalama/std'siyle z-normalize et."""
    out = df.copy()
    for sid in df["subject"].unique():
        m = df["subject"] == sid
        sub = df.loc[m, feature_cols]
        out.loc[m, feature_cols] = (sub - sub.mean()) / (sub.std() + 1e-8)
    return out


def run_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             winsorize: bool = False) -> dict[int, float]:
    """LOSO CV — RandomForest. Her denek için (test edildiğinde) accuracy döndür."""
    logo = LeaveOneGroupOut()
    per_subject: dict[int, float] = {}
    for tr, te in logo.split(X, y, groups):
        subj = int(groups[te][0])
        Xtr, Xte = X[tr].copy(), X[te].copy()

        if winsorize:
            # Kırpma sınırları SADECE train'den (leakage yok)
            lo = np.percentile(Xtr, 1, axis=0)
            hi = np.percentile(Xtr, 99, axis=0)
            Xtr = np.clip(Xtr, lo, hi)
            Xte = np.clip(Xte, lo, hi)

        pipe = make_pipeline(StandardScaler(), make_rf())
        pipe.fit(Xtr, y[tr])
        per_subject[subj] = accuracy_score(y[te], pipe.predict(Xte))
    return per_subject


def summarize(per_subject: dict[int, float]) -> tuple[float, float, float]:
    """(tüm ortalama, S13-hariç-14 ortalama, S13 fold acc) döndür."""
    all_mean = float(np.mean(list(per_subject.values())))
    others = [v for k, v in per_subject.items() if k != 13]
    others_mean = float(np.mean(others)) if others else float("nan")
    s13 = per_subject.get(13, float("nan"))
    return all_mean, others_mean, s13


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    feature_cols = [c for c in df.columns if c not in ("label", "subject", "start_sec")]
    print(f"  {len(df)} pencere, {len(feature_cols)} özellik, "
          f"{df['subject'].nunique()} denek\n")

    df_norm = per_subject_normalize(df, feature_cols)
    df_no13 = df[df["subject"] != 13].reset_index(drop=True)

    # (yöntem adı, dataframe, winsorize?)
    methods = [
        ("As-is (referans)", df, False),
        ("Per-subject norm", df_norm, False),
        ("Winsorize %1-99", df, True),
        ("Remove S13", df_no13, False),
    ]

    rows: list[dict] = []
    for task_name, make_y in [("3-sınıf", lambda d: d["label"].values.astype(np.int8)),
                              ("Binary", lambda d: (d["label"].values == 2).astype(np.int8))]:
        print(f"{'='*70}\nTASK: {task_name}\n{'='*70}")
        print(f"{'Yöntem':22s} {'Tüm ort.':>10s} {'Diğer-14':>10s} {'S13 fold':>10s}")
        for mname, mdf, winz in methods:
            X = mdf[feature_cols].values.astype(np.float32)
            y = make_y(mdf)
            groups = mdf["subject"].values
            per_subject = run_loso(X, y, groups, winsorize=winz)
            all_m, oth_m, s13 = summarize(per_subject)
            s13_str = f"{s13:.3f}" if not np.isnan(s13) else "  (yok)"
            print(f"{mname:22s} {all_m:>10.3f} {oth_m:>10.3f} {s13_str:>10s}")
            rows.append({"task": task_name, "method": mname,
                         "all_mean": all_m, "others14_mean": oth_m, "s13_fold": s13})
        print()

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS_DIR / "s13_handling_results.csv", index=False)
    print(f"[OK] CSV: outputs/s13_handling_results.csv")

    # ---- Görsel ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    method_names = [m[0] for m in methods]
    colors = ["#8E8E8E", "#4C72B0", "#55A868", "#C44E52"]
    for ax, task in zip(axes, ["3-sınıf", "Binary"]):
        sub = res[res["task"] == task].set_index("method").reindex(method_names)
        x = np.arange(len(method_names))
        width = 0.38
        ax.bar(x - width/2, sub["others14_mean"], width, label="Diğer-14 ort.",
               color=colors, edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, sub["s13_fold"], width, label="S13 fold",
               color=colors, alpha=0.45, edgecolor="black", linewidth=0.5, hatch="//")
        for xi, (o, s) in enumerate(zip(sub["others14_mean"], sub["s13_fold"])):
            ax.text(xi - width/2, o + 0.01, f"{o:.2f}", ha="center", fontsize=8, fontweight="bold")
            if not np.isnan(s):
                ax.text(xi + width/2, s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy" if task == "3-sınıf" else "")
        ax.set_title(f"{task}  (RandomForest, LOSO)")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("S13 Anomalisini Ele Alma — 4 Yöntem Karşılaştırması", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "e01_s13_handling_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Görsel: figures/e01_s13_handling_comparison.png")


if __name__ == "__main__":
    main()
