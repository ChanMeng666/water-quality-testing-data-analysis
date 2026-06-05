# Water Quality Testing Data Analysis

Statistical analysis and predictive modeling of water quality parameters using Python.

## Overview

This project analyzes a dataset of 500 water samples across five quality parameters to explore relationships between water quality indicators and build predictive models for conductivity.

**Key findings:**

- Strong positive correlation (r = 0.705) between pH and dissolved oxygen levels
- Multi-parameter linear regression model predicts conductivity from pH, temperature, turbidity, and dissolved oxygen
- OLS regression confirms statistically significant relationships between several parameter pairs (p < 0.05)

## Dataset

The dataset (`data/water_quality_testing.csv`) contains 500 samples with the following parameters:

| Parameter | Unit | Range |
|---|---|---|
| pH | pH units | 6.83 - 7.48 |
| Temperature | °C | 20.3 - 23.6 |
| Turbidity | NTU | 3.1 - 5.1 |
| Dissolved Oxygen | mg/L | 6.0 - 9.9 |
| Conductivity | µS/cm | 316 - 370 |

## Project Structure

```
water-quality-testing-data-analysis/
├── data/
│   └── water_quality_testing.csv       # Water quality dataset (500 samples)
├── notebooks/
│   └── water_quality_analysis.ipynb    # Main analysis notebook
├── .gitignore
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/ChanMeng666/water-quality-testing-data-analysis.git
cd water-quality-testing-data-analysis
pip install -r requirements.txt
```

### Usage

```bash
jupyter notebook notebooks/water_quality_analysis.ipynb
```

Run all cells (Kernel > Restart & Run All) to reproduce the full analysis.

## Analysis Contents

The notebook covers the following topics:

1. **Data Loading and Inspection** - Load dataset, examine structure and summary statistics
2. **Distribution Analysis** - Histograms with KDE for all parameters
3. **Correlation Analysis** - Correlation matrix heatmap and pair plots
4. **pH vs Dissolved Oxygen** - Deep dive into the strongest correlation
5. **Parameter Relationships** - Regression plots for multiple parameter pairs
6. **Predictive Modeling** - Linear regression for conductivity prediction (two-feature and multi-parameter models)
7. **Statistical Modeling (OLS)** - Ordinary least squares regression with statsmodels for statistical inference
8. **Conclusions** - Summary of key findings

## Built With

- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Static plotting and visualization
- [seaborn](https://seaborn.pydata.org/) - Statistical data visualization
- [scikit-learn](https://scikit-learn.org/) - Linear regression modeling
- [statsmodels](https://www.statsmodels.org/) - OLS regression and statistical testing

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Chan Meng** - [GitHub](https://github.com/ChanMeng666) · [LinkedIn](https://www.linkedin.com/in/chanmeng666/) · [Website](https://chanmeng.org/)

---

<!-- CHAN MENG PERSONAL BRAND -->
<div align="center">
  <a href="https://github.com/ChanMeng666" target="_blank">
    <img src="./.github/brand/chan-meng-logo.svg" alt="Chan Meng" width="160" />
  </a>

  <p><strong>Chan Meng</strong><br/>Need a custom app like this one? I build them — let's talk.</p>

  <a href="mailto:chanmeng.dev@gmail.com"><img src="https://img.shields.io/badge/Email-chanmeng.dev@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email Chan Meng"/></a>
  <a href="https://github.com/ChanMeng666"><img src="https://img.shields.io/badge/GitHub-ChanMeng666-181717?style=flat-square&logo=github&logoColor=white" alt="Chan Meng on GitHub"/></a>
</div>
<!-- /CHAN MENG PERSONAL BRAND -->
