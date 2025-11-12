import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dataset/tip.csv"

pd.set_option('display.max_rows', None)

def main():
    try:
        print("\n")
        
        # Loading dataset.
        data = pd.read_csv(CSV_PATH)

        # features = ["",]
        # target = "species"
        # data = data[features + [target]]
        # print(data.head())

        # data = data.dropna()
        # data = pd.get_dummies(data, drop_first=True)

        print(data)

    except KeyboardInterrupt:
        print("\nProgram has been stopped by user. (CTRL + C)")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\n")


if __name__ == "__main__":
    main()
