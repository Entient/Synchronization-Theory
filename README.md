# Synchronization-Theory
Collection of research re-analyses exploring how biological systems maintain and lose synchronization across scales — from ionic pump dynamics to heart-rate variability and systemic aging.

This repository is maintained by **[Entient LLC](https://entient.com)** and hosts data, code, and figures from ongoing studies exploring how biological systems maintain and lose synchronization across scales — from ionic pump dynamics to cardiac excitation–contraction coupling and system-level variability.  
Current focus: the **Bioelectric Coherence / Synchronization Theory of Aging**.

---

## 📁 Repository Structure
```

Synchronization-Theory/
│
├── data/                # Processed datasets, QC logs, and metadata
│   ├── Tier1/
│   ├── Tier2/
│   └── metadata/
│
├── notebooks/           # Jupyter notebooks for analysis and visualization
│   ├── 01_data_qc.ipynb
│   ├── 02_analysis_tier1.ipynb
│   ├── 03_analysis_tier2.ipynb
│   └── 04_statistics_mixed_effects.ipynb
│
├── scripts/             # Python analysis and plotting scripts
│   ├── extract_channels.py
│   ├── compute_plv_lag.py
│   └── plot_results.py
│
├── figures/             # Key output figures and schematics
│   ├── Tier1_Fig1_bandlimited_plv.png
│   ├── Tier2_Fig2_orientation_fraction.png
│   ├── Tier2_Fig3_alternans_incidence.png
│   └── schematic_3layer.png
│
├── docs/                # Extended methods, references, and summaries
│   ├── methods.md
│   ├── results_summary.md
│   └── references.bib
│
├── environment.yml      # Conda environment for reproducibility
├── LICENSE
└── README.md

````

---

## ⚙️ Environment Setup
Clone the repository and recreate the analysis environment:

```bash
git clone https://github.com/EntientLLC/Synchronization-Theory.git
cd Synchronization-Theory
conda env create -f environment.yml
conda activate sync_theory
````

---

## 🚀 Reproducing Results

1. **Quality Control** – run `notebooks/01_data_qc.ipynb`
2. **Tier-1 Analysis** – run `notebooks/02_analysis_tier1.ipynb`
3. **Tier-2 Analysis** – run `notebooks/03_analysis_tier2.ipynb`
4. **Statistics & Figures** – run `notebooks/04_statistics_mixed_effects.ipynb`

All generated figures are saved in `/figures/`.

---

## 🧠 Background

This work supports the **Synchronization Theory of Aging**, which posits that biological decline begins as loss of phase coherence between hierarchical oscillators —
from Na⁺/K⁺-ATPase pumps → cardiac rhythm → heart-rate variability.
For detailed explanation, see `docs/results_summary.md` and the accompanying manuscript.

---

## 🧾 License and Ownership

All contents © 2025 **Entient LLC**.
Released under the MIT License (see `LICENSE` for details).
If you use these materials in academic work, please credit both **Brock Richards** and **Entient LLC**.

---

## 📊 Citation

> Richards, B. et al. (2025).
> *Preserved Intracellular Coherence and Feedback-Driven Desynchronization in Cardiac Tissue.*
> [Preprint / Journal link pending]

---

## 🌐 Links

* [Entient LLC](https://entient.com)
* [Zenodo DOI (to be added)](https://zenodo.org)
* [Project Website](https://github.com/EntientLLC/Synchronization-Theory)

---

*Maintained by Entient LLC — Bioelectric Coherence Division.*
*Last updated: November 2025*

```

---

That version clearly identifies **Entient LLC** as the owner and maintainer while still leaving your individual author credit for citations.

Would you like me to create matching text for the `LICENSE` file header (so it says “Copyright © 2025 Entient LLC, released under the MIT License”)?
```
