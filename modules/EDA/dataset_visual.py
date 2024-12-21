import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def regression_visualization(df, y_test, y_pred):
    target = df.columns[-1]  
    features = df.columns[:-1]  

    min_length = min(len(df), len(y_test), len(y_pred))
    df = df.iloc[:min_length] 
    y_test = y_test[:min_length]
    y_pred = y_pred[:min_length]

    plt.figure(figsize=(8, 6))
    sns.regplot(x=np.arange(len(y_test)), y=y_test, scatter_kws={'color': 'blue', 's': 20}, line_kws={'color': 'green'})
    plt.title(f'Regression: Actual {target} Values with Trend Line')
    plt.xlabel('Index')
    plt.ylabel(target)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(y_test.values, label='Actual Values', color='blue', marker='o', markersize=4, linestyle='-')
    plt.plot(y_pred, label='Predicted Values', color='red', marker='x', markersize=4, linestyle='--')
    plt.title(f'Regression: Actual vs Predicted {target}')
    plt.xlabel('Index')
    plt.ylabel(target)
    plt.legend()
    plt.show()

def classification_visualization(df, y_test, y_pred):
    target = df.columns[-1]  

    sns.countplot(x=y_test)
    plt.title(f'Classification: Distribution of {target}')
    plt.xlabel(target)
    plt.ylabel('Count')
    plt.show()

    pairplot = sns.pairplot(df, hue=target)
    pairplot.figure.suptitle(f'Classification: Pairplot of Features (Target: {target})', y=1.02)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_pred)
    plt.title(f'Classification: Predicted Values vs Actual {target}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Count')
    plt.show()