from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor


CSV_PATH = "dataset/tip.csv"


# Visualizing Correlation (Not a must have, but just to visaulize)
def visualize_correlation(tb, t):
    # Visualize Correlation between total bill and tip.
    plt.scatter(tb, t)
    plt.xlabel("Total Bill")
    plt.ylabel("Tip")
    plt.title("Correlation between total bill and tip.")
    plt.show()


# Boxplot to check for outliers
def visualize_boxplot(data):
    plt.boxplot(data["total_bill"])
    plt.title("Boxplot of Total Bill")
    plt.ylabel("Total Bill")
    plt.show()


def main():
    try:
        print("\n")

        # Loading dataset.
        data = pd.read_csv(CSV_PATH)

        # Lets get an insight of the dataset with a row count of 10.
        print("| Insight of the dataset")
        print(f"{data.head(10)}\n\n")

        # Display NaN's
        print(f"| Not a numbers:\n{data.isna().sum()}")
        # Dropping all "Not A Number" (NaN)
        before_nans = len(data)
        data = data.dropna()
        after_nans = len(data)
        print(f"Dropped {before_nans - after_nans} rows of NaN.\n\n")

        # If there would be need for One-Hot Encodeing
        # data = pd.get_dummies(data, drop_first=True)
        # print(f"| One-Hot Encoded\n{data.head(10)}\n\n")

        # Specifying features and targets, to predict the tip size.
        X = data[["total_bill", "size"]]
        y = data["tip"]

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=67
        )

        # Initialize the Random Forest Regression model and train it
        # Overfitting
        # model = RandomForestRegressor(
        #     n_estimators=200,
        #     random_state=69,
        #     max_depth=None,
        #     min_samples_split=2,
        #     min_samples_leaf=1
        # )

        # Underfitting
        # model = RandomForestRegressor(
        #     n_estimators=10,
        #     random_state=69,
        #     max_depth=2,
        #     min_samples_split=20,
        #     min_samples_leaf=40
        # )

        model = RandomForestRegressor(
            n_estimators=240,
            random_state=69,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=10,
        )
        
        print("| Cross-validation")
        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="neg_mean_squared_error"
        )
        cv_mse_mean = -cv_scores.mean()
        cv_rmse_mean = np.sqrt(cv_mse_mean)

        print("Cross-validation RMSE (mean over 5 folds): {:.4f}\n".format(cv_rmse_mean))

        model.fit(X_train, y_train)

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
            print(f"R2 (Test): {r2_te:.4f}")
            print("-----------------------")

        print("\n| Accuracy and other Information")
        evaluate(model, X_train, X_test, y_train, y_test)
        # Feature importance (hvor meget hver feature betyder)
        importances = model.feature_importances_
        print(f"Feature Importances:")
        print(f"total_bill: {importances[0]:.4f}")
        print(f"size: {importances[1]:.4f}")
        print("\n")
        # visualize_boxplot(data)
        # visualize_correlation(data["total_bill"], data["tip"])


    # Steinmejers Error Handling
    except KeyboardInterrupt:
        print("\nProgram has been stopped by user. (CTRL + C)")
    except Exception as e:
        print(f"\nError: {e}")
        
if __name__ == "__main__":
    main()
