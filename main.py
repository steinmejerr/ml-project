from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


CSV_PATH = "dataset/tip.csv"

# Visualizing Correlation (Not a must have, but just to visaulize)
def visualizeCorrelation(tb, t):
    # Visualize Correlation between total bill and tip.
    plt.scatter(tb, t)
    plt.xlabel("Total Bill")
    plt.ylabel("Tip")
    plt.title("Correlation between total bill and tip.")
    plt.show()

def main():
    try:
        print("\n")

        # Loading dataset.
        data = pd.read_csv(CSV_PATH)
        
        print(f"Number of (NaN):\n{data.isna().sum()}")
        
        # Dropping all "Not A Number" (NaN)
        data = data.dropna()

        # Specifying features and targets, to predict the tip size.
        X = data[["total_bill", "size"]]
        y = data["tip"]

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=69
        )

        # Initialize the Linear Regression model.
        # Train it (fit)
        model = LinearRegression().fit(X_train, y_train)

        # Initialize model paramters. Calculate and add result values.
        intercept = float(model.intercept_)
        slope_tb = float(model.coef_[0])
        slope_s = float(model.coef_[1])

        # Evaluate training and testdata.
        def evaluate(model, X_tr, X_te, y_tr, y_te):
            # Predictions
            y_pred_tr = model.predict(X_tr)
            y_pred_te = model.predict(X_te)

            # Calculate MSE, RMSE and R2
            mse_tr = mean_squared_error(y_tr, y_pred_tr)
            mse_te = mean_squared_error(y_te, y_pred_te)
            rmse_tr = np.sqrt(mse_tr)
            rmse_te = np.sqrt(mse_te)
            r2_tr = r2_score(y_tr, y_pred_tr)
            r2_te = r2_score(y_te, y_pred_te)

            # Print Evaluation Output
            print(f"RMSE (Train): {rmse_tr:.4f}")
            print(f"RMSE (Test): {rmse_te:.4f}")
            print(f"R2 (Train): {r2_tr:.4f}")
            print(f"R2 (Test): {r2_te:.4f}\n")
            print("\n")

        # Print Output
        print(data.head())
        print("\n")
        print(f"Slope (Total Bill): {slope_tb:.4f}")
        print(f"Slope (Size): {slope_s:.4f}")
        print(f"Intercept: {intercept:.4f}")
        evaluate(model, X_train, X_test, y_train, y_test)
        # visualizeCorrelation(data["total_bill"], data["tip"])
        
        
        
        # Steinmejers Error Handling
    except KeyboardInterrupt:
        print("\nProgram has been stopped by user. (CTRL + C)")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\n")


if __name__ == "__main__":
    main()
