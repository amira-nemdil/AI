import pandas as pd
import numpy as np

# 1. Create a synthetic dataset
data = {
    'age': [25, 30, np.nan, 45, 50, 55, np.nan, 40],
    'income': [50000, 60000, np.nan, 80000, 100000, 120000, np.nan, 70000],
    'category': ['Red', 'Blue', 'Green', 'Red', 'Green', 'Blue', 'Blue', 'Green'],
    'education': ['High School', 'Bachelors', 'PhD', 'High School', 'PhD', 'Bachelors', 'Bachelors', 'PhD'],
    'date': ['2020-01-01', '2020/02/15', '2020/03/01', 'March 4, 2020', np.nan, '2020-05-06', '06/07/2020', 'July 8, 2020']
}

df = pd.DataFrame(data)

# 2. Handling Missing Data
# a. Remove rows with missing values
df_dropna = df.dropna()  # Creates a new DataFrame with rows containing any missing values removed
print(df_dropna)

print("---------------------------------")
from sklearn.impute import SimpleImputer, KNNImputer

# b. Impute missing values using mean for numeric features
imputer_mean = SimpleImputer(strategy='mean')  # Initialize SimpleImputer with 'mean' strategy
df['age_mean'] = imputer_mean.fit_transform(df[['age']])  # Impute missing 'age' values with the mean
df['income_mean'] = imputer_mean.fit_transform(df[['income']])  # Impute missing 'income' values with the mean
print(df['age_mean'])
print(df['income_mean'])
print("---------------------------------")

# c. Impute missing values using median for numeric features
imputer_median = SimpleImputer(strategy='median')  # Initialize SimpleImputer with 'median' strategy
df['age_median'] = imputer_median.fit_transform(df[['age']])  # Impute missing 'age' values with the median
df['income_median'] = imputer_median.fit_transform(df[['income']])  # Impute missing 'income' values with the median
print(df['age_median'])
print(df['income_median'])
print("---------------------------------")

# d. KNN imputation for missing values
knn_imputer = KNNImputer(n_neighbors=2)  # Initialize KNNImputer with 2 nearest neighbors
df_knn = pd.DataFrame(knn_imputer.fit_transform(df[['age', 'income']]), columns=['age_knn', 'income_knn'])  # Impute using KNN
print(df_knn)

print("---------------------------------")

# Step 1: Replace common separators with a standard format (replace '/' with '-')
df['date'] = df['date'].str.replace('/', '-', regex=False)  # Replace '/' with '-' in 'date' column

# Display the result
print(df[['date']])

print("---------------------------------")

# 3. Fixing Inconsistencies in Date Formats
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert 'date' to datetime, invalid parsing will be set as NaT
print(df['date'])
print("------------------------")

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder

# 4. Feature Scaling
scaler = StandardScaler()  # Initialize StandardScaler
df['income_standard'] = scaler.fit_transform(df[['income']])  # Standardize 'income'
print(df['income_standard'])

print("------------------------")

normalizer = MinMaxScaler()  # Initialize MinMaxScaler
df['income_normalized'] = normalizer.fit_transform(df[['income']])  # Normalize 'income'
print(df['income_normalized'])

print("------------------------")

robust_scaler= RobustScaler()  # Initialize RobustScaler
df['income_robust'] = robust_scaler.fit_transform(df[['income']])  # Apply robust scaling to 'income'
print(df['income_robust'])

# 5. Numeric Feature Engineering
# a. Binning: Age groups
df['age_bin'] = pd.cut(df['age_mean'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])  # Bin 'age_mean' into groups
print(df['age_bin'])
print(df['age'])

print("------------------------")

# b. Log Transformation
df['income_log'] = np.log(df['income_mean'].replace(0, np.nan))  # Apply log transformation to 'income_mean' (replace 0 with NaN to avoid errors)
print(df['income_log'])

print("------------------------")
# c. Ordinal Encoding for 'education' (e.g., High School = 1, Bachelors = 2, PhD = 3)
education_order = ['High School', 'Bachelors', 'PhD']
ordinal_encoder = OrdinalEncoder(categories=[education_order])
df['education_ordinal'] = ordinal_encoder.fit_transform(df[['education']])
print(df['education_ordinal'])
print("----------------------------")

import matplotlib.pyplot as plt
import seaborn as sns

# 7. Outlier Detection and Handling
# a. Visualizing outliers

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.boxplot(x='income', data=df)
plt.title('Box Plot for Income')

plt.subplot(1,2,2)
sns.histplot(df['income'], bins=10)
plt.title('Histogram for Income')
plt.show()

from scipy import stats
#b.Z-score methode
df['income_zscore'] = stats.zscore(df['income_mean'])
outliers_zscore = df[(df['income_zscore']>3)|(df['income_zscore'] <-3) ]

# c.IQR method for detecting outliers
Q1 = df['income_mean'].quantile(0.25)
Q3 = df['income_mean'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['income_mean'] < (Q1 -1.5 * IQR)) | (df['income_mean'] > (Q3 + 1.5 * IQR))]

# d. Removing outliers using Z-score
df_no_outliers = df [(df['income_zscore'] <= 3) & (df['income_zscore'] >= -3)]

# e. Capping outliers
df['income_capped'] = np.clip(df['income_mean'], df['income_mean'].quantile(0.01),df['income_mean'].quantile(0.99))


# 8. Feature Engineering
# a. Combining features
df['age_income_combined'] = df['age_mean'] * df['income_mean']

# b. Extracting information (e.g, exctracting year from date)
df['year'] = df['date'].dt.year

# Show final DataFrame
print(df.head())