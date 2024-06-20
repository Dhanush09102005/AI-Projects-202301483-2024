# AI-Projects-202301483-2024

Week1 : https://colab.research.google.com/drive/1Gch4ZKf_VO-NMvJdmgTeGC6CPhnmSqNm?usp=drive_link

Week2 : https://colab.research.google.com/drive/1L93BS1tONB8za91YfXKukWnbAWcporsu#scrollTo=PGjeZdk2YY9I

Task 3 :-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'titanic.csv'  
df = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe(include='all'))

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnivariate Analysis: Distribution of numerical features")
num_features = df.select_dtypes(include=[np.number])
for col in num_features.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

print("\nUnivariate Analysis: Distribution of categorical features")
cat_features = df.select_dtypes(include=[object])
for col in cat_features.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()

print("\nBivariate Analysis: Correlation heatmap of numerical features")
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

print("\nBivariate Analysis: Pairplot of numerical features")
sns.pairplot(df[num_features.columns])
plt.show()

print("\nBivariate Analysis: Categorical vs Numerical features")
for cat_col in cat_features.columns:
    for num_col in num_features.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f'{num_col} vs {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.show()

print("\nOutlier Detection: Box plots of numerical features")
for col in num_features.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(df[col])
    plt.title(f'Box plot of {col}')
    plt.xlabel(col)
    plt.show()
