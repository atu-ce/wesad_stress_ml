"""
WESAD - Hafta 1, Adım 3: Tüm denekleri 60 sn pencerelere böl, ML için tek dosya hazırla.

Karar:
  - Pencere boyutu: 60 sn
  - Adım (step): 10 sn  → 50 sn overlap (literatürde yaygın)
  - Sınıflar: 1 (Baseline) / 2 (Stress) / 3 (Amusement)
  - Sadece %100 homojen pencereleri tut (label geçişi olan pencereleri at)

Çıktı:
  outputs/windows.npz   (tek bir sıkıştırılmış dosya)
    ecg       (N, 42000)  float32  - göğüs EKG @ 700 Hz
    eda_c     (N, 42000)  float32  - göğüs EDA @ 700 Hz
    resp      (N, 42000)  float32  - göğüs solunum @ 700 Hz
    bvp       (N, 3840)   float32  - bilek PPG @ 64 Hz
    eda_w     (N, 240)    float32  - bilek EDA @ 4 Hz
    label     (N,)        int8     - 1/2/3
    subject   (N,)        int8     - 2..17
    start_sec (N,)        float32  - pencerenin başlangıç saniyesi (denek başına)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "WESAD"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CHEST_FS = 700
WRIST_FS = {"BVP": 64, "EDA": 4}

WINDOW_SEC = 60
STEP_SEC = 10
KEEP_LABELS = {1, 2, 3}

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


def window_subject(sid: int) -> list[dict]:
    """Tek deneği yükle, pencerelere böl, sadece kabul edilebilir olanları döndür."""
    pkl_path = DATA_DIR / f"S{sid}" / f"S{sid}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg = np.asarray(data["signal"]["chest"]["ECG"], dtype=np.float32).flatten()
    eda_c = np.asarray(data["signal"]["chest"]["EDA"], dtype=np.float32).flatten()
    resp = np.asarray(data["signal"]["chest"]["Resp"], dtype=np.float32).flatten()
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float32).flatten()
    eda_w = np.asarray(data["signal"]["wrist"]["EDA"], dtype=np.float32).flatten()
    labels = np.asarray(data["label"]).flatten()
    del data  # ~200 MB belleği bırak

    n_chest = WINDOW_SEC * CHEST_FS         # 42000
    n_bvp = WINDOW_SEC * WRIST_FS["BVP"]    # 3840
    n_eda_w = WINDOW_SEC * WRIST_FS["EDA"]  # 240
    step = STEP_SEC * CHEST_FS              # 7000

    results: list[dict] = []
    for start in range(0, len(ecg) - n_chest + 1, step):
        end = start + n_chest
        win_labels = labels[start:end]

        # 1. Başlangıçtaki label hedef sınıflardan biri mi?
        first_label = int(win_labels[0])
        if first_label not in KEEP_LABELS:
            continue

        # 2. Pencere boyunca label DEĞİŞMEDEN aynı mı? (homojenlik)
        if not (win_labels == first_label).all():
            continue

        # 3. Bilek senkronizasyonu: aynı saniye aralığını farklı fs'de bul
        start_sec = start / CHEST_FS
        bvp_s = int(start_sec * WRIST_FS["BVP"])
        eda_w_s = int(start_sec * WRIST_FS["EDA"])

        # 4. Bilek sinyalleri pencere kadar sığıyor mu?
        if bvp_s + n_bvp > len(bvp) or eda_w_s + n_eda_w > len(eda_w):
            continue

        results.append(
            {
                "ecg":       ecg[start:end],
                "eda_c":     eda_c[start:end],
                "resp":      resp[start:end],
                "bvp":       bvp[bvp_s:bvp_s + n_bvp],
                "eda_w":     eda_w[eda_w_s:eda_w_s + n_eda_w],
                "label":     first_label,
                "subject":   sid,
                "start_sec": start_sec,
            }
        )
    return results


def main() -> None:
    print(f"Pencere: {WINDOW_SEC} sn  |  Adım: {STEP_SEC} sn  |  Sınıflar: {sorted(KEEP_LABELS)}\n")

    all_windows: list[dict] = []
    for sid in tqdm(SUBJECT_IDS, desc="Denekler"):
        wins = window_subject(sid)
        all_windows.extend(wins)

    print(f"\n[OK] Toplam pencere: {len(all_windows)}")

    # Liste-of-dict -> numpy array stack
    ecg       = np.stack([w["ecg"] for w in all_windows])
    eda_c     = np.stack([w["eda_c"] for w in all_windows])
    resp      = np.stack([w["resp"] for w in all_windows])
    bvp       = np.stack([w["bvp"] for w in all_windows])
    eda_w     = np.stack([w["eda_w"] for w in all_windows])
    label     = np.array([w["label"] for w in all_windows], dtype=np.int8)
    subject   = np.array([w["subject"] for w in all_windows], dtype=np.int8)
    start_sec = np.array([w["start_sec"] for w in all_windows], dtype=np.float32)

    # Sınıf dağılımı
    print("\n--- Sınıf dağılımı ---")
    class_names = {1: "Baseline", 2: "Stress", 3: "Amusement"}
    total = len(label)
    for lbl in sorted(KEEP_LABELS):
        n = int((label == lbl).sum())
        print(f"  Label {lbl} ({class_names[lbl]:9s}): {n:5d}  ({100 * n / total:5.1f}%)")

    # Denek başına sayım
    print("\n--- Denek başına pencere sayısı ---")
    for sid in SUBJECT_IDS:
        mask = subject == sid
        n_total = int(mask.sum())
        n_b = int(((label == 1) & mask).sum())
        n_s = int(((label == 2) & mask).sum())
        n_a = int(((label == 3) & mask).sum())
        print(f"  S{sid:>2d}: toplam={n_total:3d}  (B={n_b}, S={n_s}, A={n_a})")

    # Kayıt
    out_path = OUTPUTS_DIR / "windows.npz"
    np.savez_compressed(
        out_path,
        ecg=ecg, eda_c=eda_c, resp=resp, bvp=bvp, eda_w=eda_w,
        label=label, subject=subject, start_sec=start_sec,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[OK] Kaydedildi: {out_path.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")

    # Belleği bırak
    del ecg, eda_c, resp, bvp, eda_w, all_windows


if __name__ == "__main__":
    main()
