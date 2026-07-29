from scripts.output import (
    decision_tree,
    random_forest,
    svm,
    ann
)


def main():

    # Sample Customer Data

    customer = [
        650,          # CreditScore
        "France",     # Geography
        "Male",       # Gender
        35,           # Age
        5,            # Tenure
        50000,        # Balance
        2,            # NumOfProducts
        1,            # HasCrCard
        1,            # IsActiveMember
        85000         # EstimatedSalary
    ]

    print("\nCustomer Details:")
    print(customer)

    print("\nPredictions")
    print("-" * 40)

    print(
        "Decision Tree :",
        decision_tree(customer)
    )

    print(
        "Random Forest :",
        random_forest(customer)
    )

    print(
        "SVM           :",
        svm(customer)
    )

    print(
        "ANN           :",
        ann(customer)
    )


if __name__ == "__main__":
    main()