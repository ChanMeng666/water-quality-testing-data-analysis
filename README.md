<div align="center">
 <h1>Water Quality Analysis Toolkit</h1>
 <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
 <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white"/>
 <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/>
 <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
 <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white"/>
</div>
<br/>

A comprehensive toolkit for analyzing and visualizing water quality data. This project provides tools and techniques for monitoring water quality parameters, performing statistical analysis, and developing predictive models for water quality assessment.

# Features

### 📊 Rich Data Visualization
Generate insightful visualizations including distribution plots, correlation matrices, and scatter plots to understand water quality parameters and their relationships.

### 📈 Statistical Analysis
Perform detailed statistical analysis including descriptive statistics, correlation analysis, and hypothesis testing on water quality parameters.

### 🤖 Predictive Modeling
Implement machine learning models to predict water quality parameters using features like pH, temperature, turbidity, and dissolved oxygen levels.

### 📉 Time Series Analysis
Analyze trends and patterns in water quality parameters over time to identify seasonal variations and long-term changes.

### 📋 Automated Reporting
Generate comprehensive reports summarizing water quality analysis results and key findings.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ChanMeng666/water-quality-testing-data-analysis.git
cd water-quality-testing-data-analysis
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels plotly
```

## Usage

1. **Load and explore the data:**
```python
import pandas as pd
water_quality_data = pd.read_csv('Water Quality Testing.csv')
print(water_quality_data.describe())
```

2. **Create visualizations:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of pH values
plt.figure(figsize=(10, 6))
sns.histplot(water_quality_data['pH'], kde=True)
plt.title('Distribution of pH Values')
plt.show()
```

3. **Perform statistical analysis:**
```python
# Correlation analysis
correlation = water_quality_data['pH'].corr(water_quality_data['Dissolved Oxygen (mg/L)'])
print('Correlation coefficient:', correlation)
```

4. **Build predictive models:**
```python
from sklearn import linear_model

# Predict conductivity using multiple parameters
X = water_quality_data[['pH', 'Temperature (°C)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)']]
y = water_quality_data['Conductivity (µS/cm)']

model = linear_model.LinearRegression()
model.fit(X, y)
```

## Tech Stack
![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/seaborn-%23323330.svg?style=for-the-badge&logo=python&logoColor=white)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
