"""
WESAD - Adım 7: ÜÇ sensörlü füzyon (ECG + EDA + Solunum) — 06'nın genişletilmiş hali.

06'da ECG + EDA (2 sensör) füzyonunu yaptık. Burada 3. sensör olarak SOLUNUM
(Respiration) ekliyoruz ve 3 farklı füzyon stratejisini kıyaslıyoruz.
(Ekstra/bonus çalışma — 06 bozulmadan ayrı script.)

3. sensör neden Solunum? Stresin bilinen üçüncü göstergesi (hızlı/düzensiz nefes),
ECG/EDA'dan farklı bir fizyolojik sistem → füzyona tamamlayıcı bilgi katar.

Görev: Binary stres tespiti (Stress vs Non-stress), RandomForest, LOSO CV.

Yaklaşımlar:
  Baseline'lar (tek sensör):  ECG-only, EDA-only, Resp-only
  1. Feature Fusion (BİRLİKTE)        : ECG+EDA+Resp özellikleri TEK modelde
  2. Decision Fusion (AYRI→BİRLEŞTİR) : 3 ayrı model → olasılıkları ortala
  3. Hybrid (HİBRİT)                  : (ECG+EDA birleşik tek model) + (Resp ayrı model)
                                        → karar noktasında olasılıkları birleştir

Çıktılar:
  outputs/fusion3_results.csv
  figures/07_three_sensor_fusion.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "features.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42

ECG_FEATURES = ["hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
                "hrv_lf", "hrv_hf", "hrv_lfhf"]          # 7
EDA_FEATURES = ["scl_mean", "scl_std", "scr_count", "scr_mean_amp"]  # 4
RESP_FEATURES = ["resp_rate", "resp_rate_std"]            # 2 (3. sensör)


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def run_single(X: np.ndarray, y: np.ndarray, groups: np.ndarray
               ) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Tek özellik bloğuyla LOSO."""
    logo = LeaveOneGroupOut()
    accs, trues, preds = [], [], []
    for tr, te in logo.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), make_rf()).fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        trues.append(y[te]); preds.append(pred)
    return accs, np.concatenate(trues), np.concatenate(preds)


def run_decision_fusion(blocks: list[np.ndarray], y: np.ndarray, groups: np.ndarray
                        ) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Her blok için AYRI model eğit, olasılıklarını ortala (soft voting).
    blocks=[ECG,EDA,Resp] → 3 model; blocks=[ECG+EDA, Resp] → 2 model (hibrit)."""
    logo = LeaveOneGroupOut()
    accs, trues, preds = [], [], []
    for tr, te in logo.split(blocks[0], y, groups):
        proba_sum, classes = None, None
        for X in blocks:
            pipe = make_pipeline(StandardScaler(), make_rf()).fit(X[tr], y[tr])
            p = pipe.predict_proba(X[te])
            proba_sum = p if proba_sum is None else proba_sum + p
            classes = pipe.classes_
        p_avg = proba_sum / len(blocks)
        pred = classes[np.argmax(p_avg, axis=1)]
        accs.append(accuracy_score(y[te], pred))
        trues.append(y[te]); preds.append(pred)
    return accs, np.concatenate(trues), np.concatenate(preds)


def evaluate(name: str, n_feat: int,
             accs: list[float], true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "method": name, "n_features": n_feat,
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_macro": float(f1_score(true, pred, average="macro")),
    }


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(INPUT_PATH)
    print(f"  ECG ({len(ECG_FEATURES)}) + EDA ({len(EDA_FEATURES)}) + "
          f"Solunum ({len(RESP_FEATURES)}) = {len(ECG_FEATURES)+len(EDA_FEATURES)+len(RESP_FEATURES)} özellik\n")

    X_ecg = df[ECG_FEATURES].values.astype(np.float32)
    X_eda = df[EDA_FEATURES].values.astype(np.float32)
    X_resp = df[RESP_FEATURES].values.astype(np.float32)
    X_ecg_eda = df[ECG_FEATURES + EDA_FEATURES].values.astype(np.float32)
    X_all = df[ECG_FEATURES + EDA_FEATURES + RESP_FEATURES].values.astype(np.float32)
    groups = df["subject"].values
    y = (df["label"].values == 2).astype(np.int8)
    baseline = 1 - y.sum() / len(y)
    print(f"BINARY — majority baseline={baseline:.3f}\n")

    n_all = len(ECG_FEATURES) + len(EDA_FEATURES) + len(RESP_FEATURES)
    rows: list[dict] = []
    # --- Tek sensör baseline'lar ---
    rows.append(evaluate("ECG-only", len(ECG_FEATURES), *run_single(X_ecg, y, groups)))
    rows.append(evaluate("EDA-only", len(EDA_FEATURES), *run_single(X_eda, y, groups)))
    rows.append(evaluate("Resp-only", len(RESP_FEATURES), *run_single(X_resp, y, groups)))
    # --- 1. Feature Fusion (birlikte) ---
    rows.append(evaluate("Feature Fusion (3 birlikte)", n_all, *run_single(X_all, y, groups)))
    # --- 2. Decision Fusion (3 ayrı → birleştir) ---
    rows.append(evaluate("Decision Fusion (3 ayrı)", n_all,
                         *run_decision_fusion([X_ecg, X_eda, X_resp], y, groups)))
    # --- 3. Hybrid (ECG+EDA birleşik | Resp ayrı) ---
    rows.append(evaluate("Hybrid (ECG+EDA | Resp)", n_all,
                         *run_decision_fusion([X_ecg_eda, X_resp], y, groups)))

    print(f"{'='*66}\nÜÇ SENSÖRLÜ FÜZYON — Binary (RandomForest, LOSO)\n{'='*66}")
    for r in rows:
        print(f"  {r['method']:30s} acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}  "
              f"macro-F1={r['f1_macro']:.3f}")
    print()

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS_DIR / "fusion3_results.csv", index=False)
    print("[OK] CSV: outputs/fusion3_results.csv")

    # ---- Görsel ----
    method_order = ["ECG-only", "EDA-only", "Resp-only",
                    "Feature Fusion (3 birlikte)", "Decision Fusion (3 ayrı)",
                    "Hybrid (ECG+EDA | Resp)"]
    labels = ["ECG\nonly", "EDA\nonly", "Resp\nonly",
              "Feature Fusion\n(3 birlikte)", "Decision Fusion\n(3 ayrı→birleştir)",
              "Hybrid\n(ECG+EDA | Resp)"]
    # baseline'lar soluk, füzyonlar canlı
    colors = ["#C44E52", "#55A868", "#DD8452", "#4C72B0", "#8172B2", "#937860"]
    sub = res.set_index("method").reindex(method_order)
    x = np.arange(len(method_order))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, sub["acc_mean"], yerr=sub["acc_std"], capsize=4,
           color=colors, edgecolor="black", linewidth=0.5)
    for xi, (v, f) in enumerate(zip(sub["acc_mean"], sub["f1_macro"])):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(xi, v - 0.06, f"F1={f:.2f}", ha="center", fontsize=8, color="white")
    ax.axvline(2.5, color="gray", linestyle=":", alpha=0.6)  # baseline | fusion ayracı
    ax.text(1.0, 0.02, "tek sensör", ha="center", fontsize=9, color="gray")
    ax.text(4.0, 0.02, "füzyon", ha="center", fontsize=9, color="gray")
    ax.axhline(baseline, color="orange", linestyle="--",
               label=f"Majority baseline ({baseline:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Üç Sensörlü Füzyon: ECG + EDA + Solunum — Binary Stres Tespiti\n(RandomForest, LOSO CV)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_three_sensor_fusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Görsel: figures/07_three_sensor_fusion.png")


if __name__ == "__main__":
    main()
