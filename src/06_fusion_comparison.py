"""
WESAD - Adım 6: ECG + EDA Sensör Füzyonu Karşılaştırması (HOCANIN PROJESİ).

Görev (ders8 s.36): "WESAD'dan ECG ve EDA özelliklerini çıkar. Stres tespiti için
bu iki veriyi farklı fusion teknikleriyle (Tek Model vs Ayrı Modeller) kıyasla."

NEDEN SADECE BINARY (Stres var / yok)?
  Başlangıçta 3-sınıf (Baseline/Stress/Amusement) denendi (bkz. ek_inceleme/).
  Orada görüldü ki Baseline ile Amusement fizyolojik olarak neredeyse aynı
  (ikisi de düşük-uyarılma) → model bu ikisini ayıramıyor, 3-sınıf ~%73'te tıkanıyor.
  Bu yüzden problem, anlamlı ve daha güvenilir olan BINARY stres tespitine indirgendi
  (Baseline+Amusement = "stresli değil", Stress = "stresli"). Stres tespiti için
  zaten istenen de budur.

4 yaklaşım × RandomForest × LOSO CV (sadece binary):
  1. ECG-only          : sadece ECG/HRV özellikleri          (tek sensör baseline)
  2. EDA-only          : sadece EDA özellikleri               (tek sensör baseline)
  3. Feature Fusion    : ECG+EDA özellikleri TEK modelde      ("Tek Model")
  4. Decision Fusion   : ayrı ECG modeli + ayrı EDA modeli,
                         olasılıkları ortalayarak birleştir   ("Ayrı Modeller")

Çıktılar:
  outputs/fusion_results.csv
  figures/06_fusion_comparison.png
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

# ECG (kalp) = göğüs EKG'sinden türeyen HRV özellikleri
ECG_FEATURES = ["hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
                "hrv_lf", "hrv_hf", "hrv_lfhf"]
# EDA (terleme) = göğüs EDA'sından türeyen özellikler
EDA_FEATURES = ["scl_mean", "scl_std", "scr_count", "scr_mean_amp"]


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def run_single(X: np.ndarray, y: np.ndarray, groups: np.ndarray
               ) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Tek özellik bloğuyla LOSO. (fold accuracy listesi, pooled_true, pooled_pred)."""
    logo = LeaveOneGroupOut()
    accs, trues, preds = [], [], []
    for tr, te in logo.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), make_rf())
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        trues.append(y[te]); preds.append(pred)
    return accs, np.concatenate(trues), np.concatenate(preds)


def run_decision_fusion(X_ecg: np.ndarray, X_eda: np.ndarray, y: np.ndarray,
                        groups: np.ndarray) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Ayrı ECG + ayrı EDA modeli; predict_proba'larını ortala (soft voting)."""
    logo = LeaveOneGroupOut()
    accs, trues, preds = [], [], []
    for tr, te in logo.split(X_ecg, y, groups):
        ecg_pipe = make_pipeline(StandardScaler(), make_rf())
        eda_pipe = make_pipeline(StandardScaler(), make_rf())
        ecg_pipe.fit(X_ecg[tr], y[tr])
        eda_pipe.fit(X_eda[tr], y[tr])

        # İki modelin olasılık tahminlerini ortala → birleşik karar
        p_ecg = ecg_pipe.predict_proba(X_ecg[te])
        p_eda = eda_pipe.predict_proba(X_eda[te])
        p_avg = (p_ecg + p_eda) / 2.0
        classes = ecg_pipe.classes_           # iki model de aynı y → aynı sınıf sırası
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
    print(f"  ECG özellikleri ({len(ECG_FEATURES)}): {ECG_FEATURES}")
    print(f"  EDA özellikleri ({len(EDA_FEATURES)}): {EDA_FEATURES}\n")

    X_ecg = df[ECG_FEATURES].values.astype(np.float32)
    X_eda = df[EDA_FEATURES].values.astype(np.float32)
    X_fusion = df[ECG_FEATURES + EDA_FEATURES].values.astype(np.float32)
    groups = df["subject"].values

    # Binary stres tespiti: Stress(2) = 1, Baseline/Amusement = 0
    y = (df["label"].values == 2).astype(np.int8)
    n_stress = int(y.sum())
    baseline = 1 - n_stress / len(y)  # majority (hep "stresli değil" demek)
    print(f"BINARY — Stres={n_stress}, Stresli değil={len(y)-n_stress}, "
          f"majority baseline={baseline:.3f}\n")

    n_fusion = len(ECG_FEATURES) + len(EDA_FEATURES)
    rows: list[dict] = []
    # 1. ECG-only
    a, t, p = run_single(X_ecg, y, groups)
    rows.append(evaluate("ECG-only", len(ECG_FEATURES), a, t, p))
    # 2. EDA-only
    a, t, p = run_single(X_eda, y, groups)
    rows.append(evaluate("EDA-only", len(EDA_FEATURES), a, t, p))
    # 3. Feature-Level Fusion ("Tek Model")
    a, t, p = run_single(X_fusion, y, groups)
    rows.append(evaluate("Feature Fusion (Tek Model)", n_fusion, a, t, p))
    # 4. Decision-Level Fusion ("Ayrı Modeller")
    a, t, p = run_decision_fusion(X_ecg, X_eda, y, groups)
    rows.append(evaluate("Decision Fusion (Ayrı Modeller)", n_fusion, a, t, p))

    print(f"{'='*64}\nBINARY STRES TESPİTİ — Sonuçlar (RandomForest, LOSO)\n{'='*64}")
    for r in rows:
        print(f"  {r['method']:34s} acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}  "
              f"macro-F1={r['f1_macro']:.3f}")
    print()

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS_DIR / "fusion_results.csv", index=False)
    print("[OK] CSV: outputs/fusion_results.csv")

    # ---- Görsel: tek panel (binary) ----
    method_order = ["ECG-only", "EDA-only",
                    "Feature Fusion (Tek Model)", "Decision Fusion (Ayrı Modeller)"]
    colors = ["#C44E52", "#55A868", "#4C72B0", "#8172B2"]
    sub = res.set_index("method").reindex(method_order)
    x = np.arange(len(method_order))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, sub["acc_mean"], yerr=sub["acc_std"], capsize=4,
           color=colors, edgecolor="black", linewidth=0.5)
    for xi, (v, f) in enumerate(zip(sub["acc_mean"], sub["f1_macro"])):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(xi, v - 0.06, f"F1={f:.2f}", ha="center", fontsize=9, color="white")
    ax.axhline(baseline, color="orange", linestyle="--",
               label=f"Majority baseline ({baseline:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["ECG\nonly", "EDA\nonly", "Feature Fusion\n(Tek Model)",
                        "Decision Fusion\n(Ayrı Modeller)"], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("ECG + EDA Sensör Füzyonu — Binary Stres Tespiti\n(RandomForest, LOSO CV)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_fusion_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Görsel: figures/06_fusion_comparison.png")


if __name__ == "__main__":
    main()
