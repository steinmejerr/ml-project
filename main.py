import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dataset/movies.csv"


def main():

    # Loading dataset.
    data = pd.read_csv(CSV_PATH)

    features = ["",]
    target = "species"
    data = data[features + [target]]
    print(data.head())

    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)

    print(data.head(20))


if __name__ == "__main__":
    main()
