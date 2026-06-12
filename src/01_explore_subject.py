"""
WESAD - Hafta 1, Adım 1: Tek deneğin verisini yükle, yapısını incele, sinyalleri görselleştir.

Hedef:
  1. S2.pkl'i belleğe yükle
  2. İçindeki sinyallerin boyutlarını, sampling rate'lerini ve label dağılımını yazdır
  3. Baseline / Stress / Amusement durumlarından 30'ar saniyelik segmentler kes
  4. Göğüs (EKG, EDA, Solunum) ve bilek (PPG, EDA) sinyallerini yan yana çizdir
  5. Görseli figures/ altına PNG olarak kaydet
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "WESAD"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CHEST_FS = 700
WRIST_FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}

LABEL_NAMES = {
    0: "Transient/Undef",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation",
    5: "Transient",
    6: "Transient",
    7: "Transient",
}
LABEL_COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green", 4: "tab:purple"}


def load_subject(subject_id: int) -> dict:
    path = DATA_DIR / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def print_summary(sid: int, data: dict) -> None:
    print(f"=== Denek S{sid} ===")
    print(f"Üst-seviye anahtarlar: {list(data.keys())}")
    print(f"Sinyal kaynakları: {list(data['signal'].keys())}")

    labels = np.asarray(data["label"]).flatten()
    duration_min = len(labels) / CHEST_FS / 60
    print(f"Toplam süre: {duration_min:.1f} dakika ({len(labels):,} örnek @ {CHEST_FS} Hz)\n")

    print("--- Göğüs (RespiBAN, hepsi 700 Hz) ---")
    for name, sig in data["signal"]["chest"].items():
        print(f"  {name:6s}: shape={np.asarray(sig).shape}")

    print("\n--- Bilek (Empatica E4, farklı Hz) ---")
    for name, sig in data["signal"]["wrist"].items():
        fs = WRIST_FS.get(name, "?")
        print(f"  {name:6s}: shape={np.asarray(sig).shape}  fs={fs} Hz")

    print("\n--- Label dağılımı ---")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = LABEL_NAMES.get(int(u), "?")
        pct = 100 * c / len(labels)
        print(f"  Label {u} ({name:14s}): {c:>8,} örnek = {c / CHEST_FS / 60:5.1f} dk  ({pct:4.1f}%)")


def find_middle_window(labels: np.ndarray, target_label: int, win_samples: int) -> tuple[int, int]:
    """Verilen label'ın bulunduğu en uzun segmentin ortasından win_samples uzunluğunda pencere döndür."""
    mask = labels == target_label
    if not mask.any():
        raise ValueError(f"Label {target_label} verisinde bulunamadı.")
    start = int(np.argmax(mask))
    end = len(mask) - int(np.argmax(mask[::-1]))
    mid = (start + end) // 2
    half = win_samples // 2
    return mid - half, mid - half + win_samples


def plot_three_states(sid: int, data: dict, window_sec: int = 30) -> Path:
    labels = np.asarray(data["label"]).flatten()
    states = [1, 2, 3]  # Baseline, Stress, Amusement

    fig, axes = plt.subplots(5, 3, figsize=(15, 11), sharex="col")

    for col, lbl in enumerate(states):
        win_start, win_end = find_middle_window(labels, lbl, window_sec * CHEST_FS)
        wrist_start_sec = win_start / CHEST_FS

        # Göğüs (700 Hz)
        t_chest = np.arange(window_sec * CHEST_FS) / CHEST_FS
        ecg = np.asarray(data["signal"]["chest"]["ECG"]).flatten()[win_start:win_end]
        eda_c = np.asarray(data["signal"]["chest"]["EDA"]).flatten()[win_start:win_end]
        resp = np.asarray(data["signal"]["chest"]["Resp"]).flatten()[win_start:win_end]

        # Bilek (kendi fs'lerinde)
        def wrist_slice(name: str) -> tuple[np.ndarray, np.ndarray]:
            fs = WRIST_FS[name]
            sig = np.asarray(data["signal"]["wrist"][name]).flatten()
            s = int(wrist_start_sec * fs)
            e = s + window_sec * fs
            return np.arange(window_sec * fs) / fs, sig[s:e]

        t_bvp, bvp = wrist_slice("BVP")
        t_eda_w, eda_w = wrist_slice("EDA")

        color = LABEL_COLORS[lbl]
        axes[0, col].plot(t_chest, ecg, color=color, linewidth=0.5)
        axes[1, col].plot(t_chest, eda_c, color=color)
        axes[2, col].plot(t_chest, resp, color=color)
        axes[3, col].plot(t_bvp, bvp, color=color, linewidth=0.7)
        axes[4, col].plot(t_eda_w, eda_w, color=color, marker="o", markersize=2, linewidth=0.8)

        axes[0, col].set_title(LABEL_NAMES[lbl], fontsize=12, fontweight="bold", color=color)

    axes[0, 0].set_ylabel("EKG\n(göğüs, 700 Hz)")
    axes[1, 0].set_ylabel("EDA\n(göğüs, 700 Hz)")
    axes[2, 0].set_ylabel("Solunum\n(göğüs, 700 Hz)")
    axes[3, 0].set_ylabel("PPG / BVP\n(bilek, 64 Hz)")
    axes[4, 0].set_ylabel("EDA\n(bilek, 4 Hz)")
    for col in range(3):
        axes[-1, col].set_xlabel("Zaman (sn)")

    fig.suptitle(
        f"WESAD - Denek S{sid}: 3 Durumda 30 sn Sinyal Karşılaştırması",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / f"01_S{sid}_signal_comparison.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    sid = 2
    data = load_subject(sid)
    print_summary(sid, data)
    out = plot_three_states(sid, data)
    print(f"\n[OK] Görselleştirme kaydedildi: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
