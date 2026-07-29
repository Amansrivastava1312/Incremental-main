import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


class FeatureEngineering:

    def create_features(self, df):

        df["BalanceSalaryRatio"] = (
            df["Balance"] /
            (df["EstimatedSalary"] + 1)
        )

        df["CreditScoreAgeRatio"] = (
            df["CreditScore"] /
            (df["Age"] + 1)
        )

        df["ProductsPerTenure"] = (
            df["NumOfProducts"] /
            (df["Tenure"] + 1)
        )

        return df
    

    def remove_correlated_features(self, df, threshold=0.90):
        """
        Remove highly correlated numerical features.
        """

        numeric_df = df.select_dtypes(include=["int64", "float64"])

        corr_matrix = numeric_df.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )

        columns_to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        df = df.drop(columns=columns_to_drop)

        print("Dropped columns:", columns_to_drop)

        return df

    def apply_pca(self, X, n_components=2):
        """
        Apply PCA for visualization.
        """

        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(X)

        print(
            "Explained Variance:",
            round(pca.explained_variance_ratio_.sum(), 2)
        )

        return X_pca

    def plot_pca(self, X_pca):
        """
        Plot PCA visualization.
        """

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1]
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Visualization")

        plt.savefig("reports/pca_plot.png")
        plt.close()