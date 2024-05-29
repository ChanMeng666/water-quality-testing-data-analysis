# Water Quality Analysis and Prediction

This repository contains code for analyzing and predicting water quality parameters using various machine learning techniques and visualization tools. The primary dataset used is the "Water Quality Testing" dataset, which includes measurements such as pH, temperature, turbidity, dissolved oxygen, and conductivity.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Visualization](#visualization)
- [Modeling](#modeling)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this code, you need to have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib plotly seaborn statsmodels
```

## Dataset

The dataset used in this project is the "Water Quality Testing" dataset, which contains the following columns:

- `Sample ID`
- `pH`
- `Temperature (°C)`
- `Turbidity (NTU)`
- `Dissolved Oxygen (mg/L)`
- `Conductivity (µS/cm)`

## Usage

1. **Load the dataset:**

    ```python
    import pandas as pd
    
    water_quality_data = pd.read_csv('/path/to/Water Quality Testing.csv')
    ```

2. **Basic data exploration:**

    ```python
    water_quality_data.describe()
    water_quality_data.info()
    ```

3. **Linear Regression Model:**

    ```python
    from sklearn import linear_model
    
    reg = linear_model.LinearRegression()
    reg.fit(water_quality_data[['pH', 'Temperature (°C)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)']], water_quality_data['Conductivity (µS/cm)'])
    
    print("Intercept:", reg.intercept_)
    print("Coefficients:", reg.coef_)
    ```

4. **Make predictions:**

    ```python
    predictions = reg.predict(water_quality_data[['pH', 'Temperature (°C)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)']])
    water_quality_data['Prediction of Conductivity (µS/cm)'] = predictions
    ```

## Visualization

The repository includes various visualizations to better understand the relationships between different water quality parameters.

1. **Pairplot using Seaborn:**

    ```python
    import seaborn as sns
    
    sns.set_style('whitegrid')
    sns.pairplot(water_quality_data, kind='scatter', height=3.5)
    plt.show()
    ```

2. **Scatter plot using Matplotlib:**

    ```python
    import matplotlib.pyplot as plt
    
    plt.scatter(x=water_quality_data['Dissolved Oxygen (mg/L)'], y=water_quality_data['Conductivity (µS/cm)'], marker='x')
    plt.xlabel('Dissolved Oxygen (mg/L)')
    plt.ylabel('Conductivity (µS/cm)')
    plt.show()
    ```

3. **Histogram of pH:**

    ```python
    sns.histplot(water_quality_data, x='pH')
    plt.show()
    ```

## Modeling

1. **Ordinary Least Squares (OLS) Regression:**

    ```python
    import statsmodels.api as sma
    import numpy as np
    
    a = water_quality_data['pH']
    b = water_quality_data['Temperature (°C)']
    a = np.array(a)
    b = np.array(b)
    a = sma.add_constant(a)
    model = sma.OLS(b, a).fit()
    print(model.summary())
    ```

2. **Linear Regression with Train/Test Split:**

    ```python
    from sklearn.model_selection import train_test_split
    
    features = ['Turbidity (NTU)', 'Dissolved Oxygen (mg/L)']
    X_train, X_test, y_train, y_test = train_test_split(water_quality_data[features], water_quality_data['Conductivity (µS/cm)'], test_size=0.2)
    
    regre = linear_model.LinearRegression()
    regre.fit(X_train, y_train)
    predictions = regre.predict(X_test)
    print("Predictions:", predictions)
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
