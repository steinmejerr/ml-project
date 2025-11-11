import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dataset/movies.csv"

def main():
    
    # Loading dataset.
    data = pd.read_csv(CSV_PATH)

    print(data.head())


if __name__ == "__main__":
    main()
