import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

def main():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=iris.feature_names[0], y=iris.feature_names[1])
    plt.title("Iris Dataset - First Two Features")
    plt.show()

if __name__ == "__main__":
    main()
