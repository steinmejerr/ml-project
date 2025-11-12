import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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

        # Steinmejers error handling
    except KeyboardInterrupt:
        print("\nProgram has been stopped by user. (CTRL + C)")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\n")


if __name__ == "__main__":
    main()
