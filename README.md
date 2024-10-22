# AI
Practical examples of how to prepare data for machine learning by cleaning, transforming, and engineering features with Python. Includes handling missing data, outlier detection, and various encoding techniques.

This project demonstrates various data preprocessing techniques using Python libraries like Pandas and Scikit-learn. It covers the following:

1. Synthetic Dataset Creation:

A synthetic dataset is created with features like age, income, category, education, and date, including some missing and inconsistent data.
2. Handling Missing Data:

Removal: Rows with missing values are removed.
Imputation: Missing values are imputed using mean, median, and KNN imputation methods.
3. Fixing Date Inconsistencies:

Inconsistent date formats are standardized and converted to datetime objects.
4. Feature Scaling:

Numeric features (income) are scaled using StandardScaler, MinMaxScaler, and RobustScaler.
5. Numeric Feature Engineering:

Binning: Age is binned into different age groups.
Log Transformation: Log transformation is applied to income.
Ordinal Encoding: Education levels are ordinally encoded.
6. Categorical Feature Encoding:

One-Hot Encoding: Categorical features (category) are one-hot encoded.
Label Encoding: Another categorical feature is label encoded.
7. Outlier Detection and Handling:

Outliers in income are visualized using box plots and histograms.
Outliers are detected using Z-score and IQR methods.
Outliers are handled by removal and capping.
8.  More Feature Engineering:

New features are created by combining existing features (age and income).
Information is extracted from the date feature (year).
Libraries Used:

Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
SciPy
This project provides a practical example of how to prepare data for machine learning tasks by addressing common data quality issues and applying various feature engineering techniques.
