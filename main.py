import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dataset/tip.csv"


def main():
    try:

        # Loading dataset.
        data = pd.read_csv(CSV_PATH)

        # features = ["",]
        # target = "species"
        # data = data[features + [target]]
        # print(data.head())

        # data = data.dropna()
        # data = pd.get_dummies(data, drop_first=True)

        print(data.head())

    except KeyboardInterrupt:
        print("\n🛑 Programmet blev stoppet af brugeren (Ctrl + C).")
    except Exception as e:
        print(f"\n⚠️ Fejl: {e}")
    finally:
        print("\n👋 Programmet er afsluttet.")


if __name__ == "__main__":
    main()
