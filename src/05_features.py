"""
WESAD - Adım 5: Filtrelenmiş pencerelerden öznitelik vektörü çıkar.

Her 60 sn pencere → 18 sayılık feature vector.
Klasik ML modelleri (RF, SVM, XGBoost) bu tabloyu doğrudan kullanır.

Öznitelikler (Ders 5'in pratik karşılığı):
  EKG / HRV (7):
    - hr_mean      : ortalama kalp hızı (BPM)
    - hrv_sdnn     : RR aralıklarının standart sapması (ms)
    - hrv_rmssd    : ardışık RR farklarının RMS'i (ms, parasempatik gösterge)
    - hrv_pnn50    : ardışık RR farkı > 50ms olan oran (%)
    - hrv_lf       : Low Frequency (0.04-0.15 Hz) güç (sempatik+parasempatik karışık)
    - hrv_hf       : High Frequency (0.15-0.40 Hz) güç (parasempatik baskın)
    - hrv_lfhf     : LF/HF oranı (sempato-vagal denge)

  EDA göğüs (4):
    - scl_mean     : ortalama tonik seviye (µS)
    - scl_std      : tonik seviye standart sapması
    - scr_count    : fazik tepe sayısı (uyarılma sayacı)
    - scr_mean_amp : tepelerin ortalama genliği

  Solunum (2):
    - resp_rate     : nefes/dk
    - resp_rate_std : nefes hızının değişkenliği

  BVP bilek (2):
    - bvp_hr_mean   : PPG'den ortalama kalp hızı
    - bvp_hrv_sdnn  : PPG'den HRV (göğüse paralel ölçü)

  EDA bilek (3):
    - scl_w_mean   : ortalama
    - scl_w_std    : standart sapma
    - scl_w_slope  : pencere boyunca eğim (yükseliyor mu, düşüyor mu?)

Çıktı:
  outputs/features.csv  -  3040 satır × 21 sütun (18 özellik + label + subject + start_sec)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")  # neurokit2 ve scipy bilgilendirme mesajları

from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "windows_filtered.npz"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "features.csv"

CHEST_FS = 700
WRIST_FS = {"BVP": 64, "EDA": 4}

ECG_KEYS = ["hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf", "hrv_hf", "hrv_lfhf"]
EDA_C_KEYS = ["scl_mean", "scl_std", "scr_count", "scr_mean_amp"]
RESP_KEYS = ["resp_rate", "resp_rate_std"]
BVP_KEYS = ["bvp_hr_mean", "bvp_hrv_sdnn"]
EDA_W_KEYS = ["scl_w_mean", "scl_w_std", "scl_w_slope"]

ALL_FEATURE_KEYS = ECG_KEYS + EDA_C_KEYS + RESP_KEYS + BVP_KEYS + EDA_W_KEYS


def _nan_dict(keys: list[str]) -> dict:
    return {k: np.nan for k in keys}


def extract_ecg(ecg: np.ndarray, fs: int = CHEST_FS) -> dict:
    """Time + frequency domain HRV. R-peak detection NeuroKit2 ile."""
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs, correct_artifacts=True)
        peaks = info["ECG_R_Peaks"]
        if len(peaks) < 5:
            return _nan_dict(ECG_KEYS)

        rr_ms = np.diff(peaks) / fs * 1000.0  # RR intervalleri ms cinsinden

        # Time-domain
        hr_mean = 60_000.0 / rr_ms.mean()
        sdnn = float(np.std(rr_ms, ddof=1))
        diffs = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        pnn50 = float((np.abs(diffs) > 50).mean() * 100)

        # Frequency-domain: RR serisini düzgün ızgaraya yeniden örnekle, sonra PSD
        rr_times_s = np.cumsum(rr_ms) / 1000.0
        if rr_times_s[-1] < 30:
            lf = hf = lfhf = np.nan
        else:
            fs_rr = 4  # Hz, RR sinyali için yeterli
            t_uniform = np.arange(0, rr_times_s[-1], 1.0 / fs_rr)
            rr_uniform = np.interp(t_uniform, rr_times_s, rr_ms)
            freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=min(256, len(rr_uniform)))
            lf = float(psd[(freqs >= 0.04) & (freqs < 0.15)].sum())
            hf = float(psd[(freqs >= 0.15) & (freqs < 0.40)].sum())
            lfhf = lf / hf if hf > 0 else np.nan

        return {
            "hr_mean": hr_mean, "hrv_sdnn": sdnn, "hrv_rmssd": rmssd, "hrv_pnn50": pnn50,
            "hrv_lf": lf, "hrv_hf": hf, "hrv_lfhf": lfhf,
        }
    except Exception:
        return _nan_dict(ECG_KEYS)


def extract_eda_chest(eda: np.ndarray, fs: int = CHEST_FS) -> dict:
    """Tonik (SCL) + fazik (SCR) öznitelikler."""
    try:
        scl_mean = float(np.mean(eda))
        scl_std = float(np.std(eda))

        _, info = nk.eda_peaks(eda, sampling_rate=fs)
        peaks = info.get("SCR_Peaks", [])
        amps = info.get("SCR_Amplitude", [])

        scr_count = int(len(peaks))
        scr_mean_amp = float(np.nanmean(amps)) if scr_count > 0 else 0.0

        return {"scl_mean": scl_mean, "scl_std": scl_std,
                "scr_count": scr_count, "scr_mean_amp": scr_mean_amp}
    except Exception:
        return _nan_dict(EDA_C_KEYS)


def extract_resp(resp: np.ndarray, fs: int = CHEST_FS) -> dict:
    """Nefes hızı (peak-tabanlı sayım)."""
    try:
        # Min mesafe: 2 sn → max 30 nefes/dk
        peaks, _ = find_peaks(resp, distance=int(fs * 2.0))
        if len(peaks) < 3:
            return _nan_dict(RESP_KEYS)
        intervals_s = np.diff(peaks) / fs
        rates = 60.0 / intervals_s
        return {"resp_rate": float(rates.mean()), "resp_rate_std": float(rates.std())}
    except Exception:
        return _nan_dict(RESP_KEYS)


def extract_bvp(bvp: np.ndarray, fs: int = WRIST_FS["BVP"]) -> dict:
    """PPG sistolik peak'lerinden HR ve basit HRV."""
    try:
        # Min mesafe: 0.4 sn → max 150 BPM
        peaks, _ = find_peaks(bvp, distance=int(fs * 0.4))
        if len(peaks) < 5:
            return _nan_dict(BVP_KEYS)
        rr_ms = np.diff(peaks) / fs * 1000.0
        return {"bvp_hr_mean": float(60_000.0 / rr_ms.mean()),
                "bvp_hrv_sdnn": float(np.std(rr_ms, ddof=1))}
    except Exception:
        return _nan_dict(BVP_KEYS)


def extract_eda_wrist(eda: np.ndarray, fs: int = WRIST_FS["EDA"]) -> dict:
    """Düşük örnekleme — sadece ortalama, std, trend (eğim)."""
    try:
        mean = float(np.mean(eda))
        std = float(np.std(eda))
        t = np.arange(len(eda)) / fs
        slope = float(np.polyfit(t, eda, deg=1)[0])  # µS/sn (yükseliş hızı)
        return {"scl_w_mean": mean, "scl_w_std": std, "scl_w_slope": slope}
    except Exception:
        return _nan_dict(EDA_W_KEYS)


def main() -> None:
    print(f"Yükleniyor: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    npz = np.load(INPUT_PATH)
    ecg = npz["ecg"]; eda_c = npz["eda_c"]; resp = npz["resp"]
    bvp = npz["bvp"]; eda_w = npz["eda_w"]
    label = npz["label"]; subject = npz["subject"]; start_sec = npz["start_sec"]
    n = len(label)
    print(f"  {n} pencere\n")

    print("Öznitelik çıkarımı (her pencere için EKG/EDA/Resp/BVP/EDA-w):")
    rows: list[dict] = []
    for i in tqdm(range(n)):
        feat: dict = {}
        feat.update(extract_ecg(ecg[i]))
        feat.update(extract_eda_chest(eda_c[i]))
        feat.update(extract_resp(resp[i]))
        feat.update(extract_bvp(bvp[i]))
        feat.update(extract_eda_wrist(eda_w[i]))
        feat["label"] = int(label[i])
        feat["subject"] = int(subject[i])
        feat["start_sec"] = float(start_sec[i])
        rows.append(feat)

    df = pd.DataFrame(rows)
    df = df[ALL_FEATURE_KEYS + ["label", "subject", "start_sec"]]

    # NaN raporu
    print("\n--- NaN oranları (başarısız extraction) ---")
    any_nan = False
    for col in ALL_FEATURE_KEYS:
        nan_pct = 100 * df[col].isna().mean()
        if nan_pct > 0:
            print(f"  {col:14s}: {nan_pct:5.1f}% NaN")
            any_nan = True
    if not any_nan:
        print("  Hiçbir sütunda NaN yok [OK]")

    # Sınıf başına özet
    print("\n--- Sınıf başına ortalamalar ---")
    name_map = {1: "Baseline", 2: "Stress", 3: "Amusement"}
    summary = df.groupby("label")[ALL_FEATURE_KEYS].mean().T
    summary.columns = [name_map[c] for c in summary.columns]
    print(summary.round(2).to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Kaydedildi: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}  ({len(df)} satır × {len(df.columns)} sütun)")


if __name__ == "__main__":
    main()
