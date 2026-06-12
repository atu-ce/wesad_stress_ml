"""
WESAD - Adım 4: windows.npz'deki ham sinyalleri filtrele.

Filtreler (her sinyalin fizyolojik bandına göre):
  - EKG (700 Hz)        : Band-pass 0.5-40 Hz + Notch 50 Hz   (drift + kas + şebeke)
  - EDA göğüs (700 Hz)  : Low-pass 5 Hz                       (yüksek-freq gürültü)
  - Solunum (700 Hz)    : Band-pass 0.1-0.35 Hz               (sadece nefes bandı)
  - BVP bilek (64 Hz)   : Band-pass 0.5-8 Hz                  (PPG temel + harmonik)
  - EDA bilek (4 Hz)    : Low-pass 1 Hz                       (Nyquist 2 Hz, hafif)

Tüm filtreler ZERO-PHASE (sosfiltfilt) — sinyal şekli korunur,
sadece istenmeyen frekanslar bastırılır. R-tepe konumları kaymaz.

Çıktı:
  outputs/windows_filtered.npz   (aynı yapı, temizlenmiş)
  figures/04_filter_before_after.png (1 pencere/sınıf, ham vs filtrelenmiş)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "windows.npz"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "windows_filtered.npz"
FIGURES_DIR = PROJECT_ROOT / "figures"

CHEST_FS = 700
WRIST_FS = {"BVP": 64, "EDA": 4}


def build_filters() -> dict[str, list]:
    """Tüm filtreleri tek seferde tasarla — pencere başına yeniden hesaplama yok."""
    filters: dict[str, list] = {}

    # EKG: önce band-pass, sonra 50 Hz notch
    sos_ecg_bp = butter(4, [0.5, 40], btype="band", fs=CHEST_FS, output="sos")
    b_n, a_n = iirnotch(50.0, Q=30.0, fs=CHEST_FS)
    sos_ecg_notch = tf2sos(b_n, a_n)
    filters["ecg"] = [sos_ecg_bp, sos_ecg_notch]

    # EDA göğüs: yavaş sinyal, sadece düşük frekanslar
    filters["eda_c"] = [butter(4, 5, btype="low", fs=CHEST_FS, output="sos")]

    # Solunum: dar band-pass (6-21 nefes/dk = 0.1-0.35 Hz)
    filters["resp"] = [butter(2, [0.1, 0.35], btype="band", fs=CHEST_FS, output="sos")]

    # BVP: PPG bandı
    filters["bvp"] = [butter(4, [0.5, 8], btype="band", fs=WRIST_FS["BVP"], output="sos")]

    # EDA bilek: çok düşük örnekleme, hafif low-pass
    filters["eda_w"] = [butter(2, 1, btype="low", fs=WRIST_FS["EDA"], output="sos")]

    return filters


def apply_filters(signal: np.ndarray, sos_list: list[np.ndarray]) -> np.ndarray:
    """SOS filtrelerini sırayla, vectorized + zero-phase olarak uygula."""
    out = signal.astype(np.float64)  # filtfilt için 64-bit daha kararlı
    for sos in sos_list:
        out = sosfiltfilt(sos, out, axis=-1)
    return out.astype(np.float32)


def plot_before_after(raw: dict, filt: dict, label: np.ndarray, out_path: Path) -> None:
    """Her sınıftan bir pencere için EKG/EDA-c/Solunum ham vs filtrelenmiş."""
    state_idx = {
        1: int(np.argmax(label == 1)),
        2: int(np.argmax(label == 2)),
        3: int(np.argmax(label == 3)),
    }
    state_names = {1: "Baseline", 2: "Stress", 3: "Amusement"}
    state_colors = {1: "#4C72B0", 2: "#C44E52", 3: "#55A868"}

    channels = [
        ("ecg", CHEST_FS, "EKG (göğüs)"),
        ("eda_c", CHEST_FS, "EDA (göğüs)"),
        ("resp", CHEST_FS, "Solunum"),
    ]

    fig, axes = plt.subplots(len(channels), 3, figsize=(15, 9), sharex="col")
    for col, lbl in enumerate([1, 2, 3]):
        idx = state_idx[lbl]
        for row, (key, fs, title) in enumerate(channels):
            ax = axes[row, col]
            t = np.arange(raw[key].shape[1]) / fs
            ax.plot(t, raw[key][idx], color="lightgray", linewidth=0.6, label="Ham")
            ax.plot(t, filt[key][idx], color=state_colors[lbl], linewidth=0.8, label="Filtrelenmiş")
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(state_names[lbl], color=state_colors[lbl], fontweight="bold")
            if row == len(channels) - 1:
                ax.set_xlabel("Zaman (sn)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.2)

    fig.suptitle("Filtre Öncesi (gri) vs Sonrası (renkli) — 1 pencere / sınıf", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    npz = np.load(INPUT_PATH)
    raw = {key: npz[key] for key in ["ecg", "eda_c", "resp", "bvp", "eda_w"]}
    label = npz["label"]
    subject = npz["subject"]
    start_sec = npz["start_sec"]
    print(f"  {len(label)} pencere yüklendi.\n")

    print("Filtreler tasarlanıyor (Butterworth + Notch, sos form)...")
    filters = build_filters()

    print("\nFiltreler uygulanıyor (zero-phase, vectorized):")
    filt: dict[str, np.ndarray] = {}
    for key in ["ecg", "eda_c", "resp", "bvp", "eda_w"]:
        print(f"  {key:6s}  shape={raw[key].shape}", end="  ... ", flush=True)
        filt[key] = apply_filters(raw[key], filters[key])
        print("OK")

    print(f"\nKaydediliyor: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    np.savez_compressed(
        OUTPUT_PATH,
        ecg=filt["ecg"], eda_c=filt["eda_c"], resp=filt["resp"],
        bvp=filt["bvp"], eda_w=filt["eda_w"],
        label=label, subject=subject, start_sec=start_sec,
    )
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"  Dosya boyutu: {size_mb:.1f} MB")

    fig_path = FIGURES_DIR / "04_filter_before_after.png"
    print(f"\nKarşılaştırma görseli üretiliyor: {fig_path.relative_to(PROJECT_ROOT)}")
    plot_before_after(raw, filt, label, fig_path)
    print("[OK] Bitti.")


if __name__ == "__main__":
    main()
