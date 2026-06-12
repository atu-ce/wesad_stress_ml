"""
WESAD - Hafta 1, Adım 2: 15 deneğin tümünü tara, demografi ve label dağılımlarını topla.

Hedef:
  1. Her deneğin .pkl'inden label dizisini al (sinyalleri yüklemeden, hafıza tasarrufu)
  2. Her deneğin readme.txt'sinden yaş/cinsiyet/boy/kilo/notlar parse et
  3. Tüm bilgiyi tek bir pandas DataFrame'de topla
  4. CSV olarak outputs/ altına kaydet
  5. İki grafik üret:
      a) Her deneğin label dağılımı (stacked bar)
      b) Demografi özeti (yaş histogramı + cinsiyet pie)
  6. Veri-kalite uyarılarını yazdır (kısa stres bloğu, eksik veri vs.)
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "WESAD"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CHEST_FS = 700
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]  # S1, S12 yok

LABEL_NAMES = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
LABEL_COLORS = {1: "#4C72B0", 2: "#C44E52", 3: "#55A868", 4: "#8172B2"}


def parse_readme(path: Path) -> dict:
    """readme.txt'den demografi ve notları ayıkla. Esnek regex parse — eksik alanlara dayanıklı."""
    text = path.read_text(encoding="utf-8", errors="replace")
    info: dict = {}

    patterns = {
        "age": r"Age:\s*(\d+)",
        "height_cm": r"Height\s*\(cm\):\s*(\d+)",
        "weight_kg": r"Weight\s*\(kg\):\s*(\d+)",
        "gender": r"Gender:\s*(\w+)",
        "hand": r"Dominant hand:\s*(\w+)",
        "coffee_today": r"drink coffee today\?\s*(\w+)",
        "smoker": r"Are you a smoker\?\s*(\w+)",
        "ill": r"feel ill today\?\s*(\w+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1)
            info[key] = int(val) if val.isdigit() else val.lower()

    notes_match = re.search(r"### Additional notes ###\s*(.+)", text, re.DOTALL)
    info["notes"] = notes_match.group(1).strip() if notes_match else ""
    return info


def summarize_subject(sid: int) -> dict:
    """Tek bir deneğin özetini çıkar."""
    pkl_path = DATA_DIR / f"S{sid}" / f"S{sid}.pkl"
    readme_path = DATA_DIR / f"S{sid}" / f"S{sid}_readme.txt"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    labels = np.asarray(data["label"]).flatten()
    del data  # 200+ MB belleği hemen bırak

    row: dict = {"subject": f"S{sid}"}
    if readme_path.exists():
        row.update(parse_readme(readme_path))

    row["total_min"] = round(len(labels) / CHEST_FS / 60, 2)
    for lbl, name in LABEL_NAMES.items():
        n = int((labels == lbl).sum())
        row[f"{name.lower()}_min"] = round(n / CHEST_FS / 60, 2)
    row["transient_min"] = round(((labels == 0) | (labels >= 5)).sum() / CHEST_FS / 60, 2)
    return row


def plot_label_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Her deneğin label dağılımını stacked bar olarak çiz."""
    fig, ax = plt.subplots(figsize=(13, 6))
    cols = ["baseline_min", "stress_min", "amusement_min", "meditation_min"]
    labels = ["Baseline", "Stress", "Amusement", "Meditation"]
    colors = [LABEL_COLORS[1], LABEL_COLORS[2], LABEL_COLORS[3], LABEL_COLORS[4]]

    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for col, name, color in zip(cols, labels, colors):
        ax.bar(x, df[col], bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.5)
        bottom += df[col].values

    ax.set_xticks(x)
    ax.set_xticklabels(df["subject"], rotation=0)
    ax.set_ylabel("Süre (dakika)")
    ax.set_xlabel("Denek")
    ax.set_title("WESAD - Denek başına label dağılımı (sadece kullanılacak durumlar)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_demographics(df: pd.DataFrame, out_path: Path) -> None:
    """Demografi: yaş histogramı + cinsiyet pie chart."""
    fig, (ax_age, ax_gender) = plt.subplots(1, 2, figsize=(11, 4))

    ax_age.hist(df["age"].dropna(), bins=range(20, 40, 2), color="#4C72B0", edgecolor="black")
    ax_age.set_xlabel("Yaş")
    ax_age.set_ylabel("Denek sayısı")
    ax_age.set_title(
        f"Yaş dağılımı (ort={df['age'].mean():.1f}, min={df['age'].min():.0f}, max={df['age'].max():.0f})"
    )
    ax_age.grid(axis="y", alpha=0.3)

    counts = df["gender"].value_counts()
    ax_gender.pie(
        counts.values,
        labels=[f"{g} ({n})" for g, n in counts.items()],
        colors=["#55A868", "#C44E52"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax_gender.set_title("Cinsiyet dağılımı")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def warn_data_quality(df: pd.DataFrame) -> list[str]:
    """Veri-kalite uyarıları."""
    warnings: list[str] = []
    median_stress = df["stress_min"].median()
    for _, row in df.iterrows():
        sid = row["subject"]
        # Stres bloğu beklenenden çok kısaysa
        if row["stress_min"] < median_stress * 0.7:
            warnings.append(
                f"  ! {sid}: stres süresi {row['stress_min']:.1f} dk "
                f"(median={median_stress:.1f} dk) - beklenenden kısa"
            )
        # Toplam süre 80 dakikadan azsa
        if row["total_min"] < 80:
            warnings.append(f"  ! {sid}: toplam süre {row['total_min']:.1f} dk - beklenenden kısa")
        # readme'de uyarıcı not varsa
        notes = row.get("notes", "")
        if notes and any(kw in notes.lower() for kw in ["not", "issue", "problem", "fail", "loose"]):
            warnings.append(f"  ! {sid}: readme notu - {notes[:80]}...")
    return warnings


def main() -> None:
    print(f"=== {len(SUBJECT_IDS)} denek taranıyor ===\n")
    rows = []
    for sid in SUBJECT_IDS:
        print(f"  S{sid} yükleniyor...", end=" ", flush=True)
        row = summarize_subject(sid)
        rows.append(row)
        print(f"OK ({row['total_min']:.1f} dk)")

    df = pd.DataFrame(rows)
    cols_order = [
        "subject", "age", "gender", "height_cm", "weight_kg", "hand",
        "coffee_today", "smoker", "ill",
        "total_min", "baseline_min", "stress_min", "amusement_min", "meditation_min", "transient_min",
        "notes",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    csv_path = OUTPUTS_DIR / "subjects_overview.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV: {csv_path.relative_to(PROJECT_ROOT)}")

    fig1 = FIGURES_DIR / "02_all_subjects_label_distribution.png"
    plot_label_distribution(df, fig1)
    print(f"[OK] Görsel 1: {fig1.relative_to(PROJECT_ROOT)}")

    fig2 = FIGURES_DIR / "02_all_subjects_demographics.png"
    plot_demographics(df, fig2)
    print(f"[OK] Görsel 2: {fig2.relative_to(PROJECT_ROOT)}")

    print("\n=== Özet istatistikler ===")
    print(f"Toplam denek: {len(df)}")
    print(f"Toplam kayıt süresi: {df['total_min'].sum():.0f} dk = {df['total_min'].sum() / 60:.1f} saat")
    print(f"Yaş: ortalama={df['age'].mean():.1f}, std={df['age'].std():.1f}, aralık={df['age'].min():.0f}-{df['age'].max():.0f}")
    print(f"Cinsiyet: {df['gender'].value_counts().to_dict()}")
    print()
    print("Sınıf başına TOPLAM süre (15 denek birleşik):")
    for col, name in [("baseline_min", "Baseline"), ("stress_min", "Stress"),
                      ("amusement_min", "Amusement"), ("meditation_min", "Meditation")]:
        total = df[col].sum()
        share = 100 * total / (df["baseline_min"].sum() + df["stress_min"].sum()
                                + df["amusement_min"].sum() + df["meditation_min"].sum())
        print(f"  {name:12s}: {total:6.1f} dk  ({share:4.1f}% kullanılabilir verinin)")

    warnings = warn_data_quality(df)
    if warnings:
        print("\n=== Veri-kalite uyarıları ===")
        for w in warnings:
            print(w)
    else:
        print("\nVeri-kalite uyarısı yok.")


if __name__ == "__main__":
    main()
