import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from scipy import stats
import numpy as np

def remove_outliers(df):
    print("Choose an outlier removal method:")
    print("1. Z-Score")
    print("2. IQR")
    print("3. KNN (Local Outlier Factor)")
    print("4. DBSCAN")

    method_choice = input("Choose a method (1-4): ")

    column = input("Enter the column name to check for outliers: ")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot Before {method_choice} Removal')

    plt.show()

    confirm = input("Do you see outliers? Do you want to remove them? (y/n): ")
    if confirm.lower() != 'y':
        print("Outlier removal cancelled.")
        return df

    if method_choice == '1': 
        threshold = float(input("Enter the Z-Score threshold (e.g., 3): "))
        z_scores = np.abs(stats.zscore(df[column]))
        df = df[z_scores < threshold]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[column])
        plt.title('Box Plot After Z-Score Removal')

    elif method_choice == '2':  
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[column])
        plt.title('Box Plot After IQR Removal')

    elif method_choice == '3':  
        lof = LocalOutlierFactor(n_neighbors=20)
        outliers = lof.fit_predict(df[[column]])
        df = df[outliers == 1]  

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=df.index, y=df[column])
        plt.title('Scatter Plot After KNN Removal')

    elif method_choice == '4':  
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        outliers = dbscan.fit_predict(df[[column]])
        df = df[outliers != -1]  

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=df.index, y=df[column])
        plt.title('Scatter Plot After DBSCAN Removal')

    else:
        print("Invalid choice.")
        return df

    plt.tight_layout()
    plt.show()
    print(f"Outliers removed from column: {column}")
    
    return df