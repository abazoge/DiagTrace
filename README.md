# Prompting Language Influences Diagnostic Reasoning and Accuracy of Large Language Models

This repository contains the scores data, analysis scripts, and figure generation code for the paper:

> **Prompting language influences diagnostic reasoning and accuracy of large language models**
>
> Adrien Bazoge, Josselin Corvellec, Sofiane Sid-Ahmed, Pierre-Antoine Gourraud. 2025.

## Overview

We evaluated five large language models (o3, DeepSeek-R1, GPT-4-Turbo, Llama-3.1-405B-Instruct, and BioMistral-7B) on 180 bilingual clinical vignettes (English and French) across 16 medical specialties. Model outputs were independently assessed by two general practitioners using an 18-point evaluation scale covering diagnostic reasoning quality and final diagnosis accuracy.

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── data.xlsx              # General practitioners evaluation scores
├── scripts/
│   ├── utils.py               # Shared utilities and data loading functions
│   ├── main_results.R         # Main results: Tables 1-2, Supplementary Tables S2-S4
│   ├── inter_rater_agreement.py  # Inter-rater agreement: Supplementary Table S5
│   ├── sensitivity_analysis.R # Sensitivity analysis: Supplementary Tables S6-S7
│   ├── bubble_plot.py         # Figure 1: Bubble plots (EN vs FR pairwise comparisons)
│   ├── forest_plot.py         # Figure 2: Forest plot (LMM effect sizes)
│   ├── radar_plot.py          # Figure 3: Radar plots (performance by evaluation criteria)
│   └── hist_plot.py           # Supplementary Figure S1: Score distributions
├── figures/                   # Generated figures (.png, .pdf)
└── tables/                    # Generated tables (.tsv)
```

## Requirements

### R (version 4.5.1)
- readxl
- dplyr
- tidyr
- lme4
- lmerTest

### Python (version 3.11)
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- openpyxl
- seaborn

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install R dependencies:
```R
install.packages(c("readxl", "dplyr", "tidyr", "lme4", "lmerTest"))
```

## Reproducing Results

### Tables

Generate the main results tables (Tables 1-2) and supplementary tables (S2-S4):
```bash
cd scripts
Rscript main_results.R
```

Generate the sensitivity analysis tables (S6-S7):
```bash
cd scripts
Rscript sensitivity_analysis.R
```

Generate the inter-rater agreement table (S5):
```bash
cd scripts
python inter_rater_agreement.py
```

All tables are saved as `.tsv` files in the `tables/` directory.

### Figures

Generate all figures:
```bash
cd scripts
python bubble_plot.py       # Figure 1
python forest_plot.py       # Figure 2
python radar_plot.py        # Figure 3
python hist_plot.py         # Supplementary Figure S1
```

All figures are saved in the `figures/` directory.

## Data

The dataset in this repository (`data/data.xlsx`) contains only general practitioners evaluation scores for all five models across six criteria.
The complete dataset (180 bilingual clinical vignettes with metadata and evaluation scores) is available on Hugging Face (https://huggingface.co/datasets/ANR-MALADES/DiagTrace).

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{,
  title={Prompting language influences diagnostic reasoning and accuracy of large language models},
  author={Adrien Bazoge and Josselin Corvellec and Sofiane Sid-Ahmed and Pierre-Antoine Gourraud},
  year={2026},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={}, 
}
```

## License

The code in this repository is licensed under the [MIT License](LICENSE).

The dataset (`data/data.xlsx`) is licensed under 
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgements

This work was financially supported by ANR MALADES (ANR-23-IAS1-0005). Computational resources were provided by GENCI-IDRIS (Grant 2024-AD011013715R2).