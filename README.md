### London Crime Data Analysis

This project focuses on analyzing the **London Crime Dataset** using data clustering and linear fitting techniques. The goal is to discover hidden patterns and trends in crime across London boroughs.

---

### 📁 Dataset

The analysis is based on the **London Crime Dataset**, which contains crime reports from 2008 to 2016. The dataset typically includes the following columns:

- `borough`: Name of the London borough.
- `major_category`: Broad category of the crime (e.g., Theft, Violence).
- `minor_category`: Specific type of crime.
- `value`: Number of crimes reported.
- `year`: Year of the crime report.
- `month`: Month of the crime report.

> **Note**: Ensure your dataset is preprocessed before running the analysis.

---

### 🧪 Features and Analysis

The script includes the following features:

#### 1. **Data Preprocessing**
- Aggregation and reshaping of crime values per borough.
- Normalization of data to prepare for clustering.

#### 2. **K-Means Clustering**
- Uses KMeans to cluster boroughs based on crime rate patterns.
- Elbow method is used to determine the optimal number of clusters.

#### 3. **Visualization**
- Boroughs are visualized in a scatter plot based on crime characteristics.
- Clusters are represented with different colors.

#### 4. **Linear Regression Fitting**
- Fits a line of best fit using linear regression.
- Compares predicted vs. actual crime trends.

---

### 🧰 Requirements

Make sure the following libraries are installed:

```bash
numpy
pandas
matplotlib
scikit-learn
```

#### Install via pip if needed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

### 🚀 How to Run

Run the analysis script using:

```bash
python clustering_and_fitting.py
```

Make sure the dataset CSV file is available in the same directory or update the path accordingly inside the script.

---

### 📊 Output

The script generates:

- Cluster visualizations.
- Plots showing the linear regression fitting of crimes over time.
- Printed metrics and analysis on clustering and fitting.

---

### 📌 Notes

- Clustering is sensitive to data scaling, so normalization is applied.
- The script assumes a specific structure of the input dataset—adjust if your columns vary.
- Linear regression is used here for basic trend analysis. More advanced time-series techniques may improve forecasting.

---

### 👨‍💼 Author

This project was developed for exploring machine learning and data analysis on real-world urban data.

