import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # Loading dataset.
    data = sns.load_dataset("penguins")

    print(data.head())
    # print(data.info())
    # print(data.describe())
    # print(data.isna().sum())


if __name__ == "__main__":
    main()
