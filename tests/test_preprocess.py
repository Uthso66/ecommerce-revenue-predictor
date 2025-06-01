import pandas as pd
from src.data.preprocess_data import split_data

def test_split_data():
    df = pd.DataFrame({
        'Avg. Session Length': [1, 2, 3, 4],
        'Time on App': [1, 2, 3, 4],
        'Time on Website': [1, 2, 3, 4],
        'Length of Membership': [1, 2, 3, 4],
        'Email': ['a', 'b', 'c', 'd'],
        'Address': ['x', 'y', 'z', 'w'],
        'Avatar': ['img1', 'img2', 'img3', 'img4'],
        'Yearly Amount Spent': [10, 20, 30, 40]
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, 0.25, 0.25, 0)
    assert len(X_train) + len(X_val) + len(X_test) == 4
    assert 'Email' not in X_train.columns

    print(f"\n🌈 Split Visualization:")
    print(f"Train: {'🟦' * len(X_train)}")
    print(f"Val:   {'🟩' * len(X_val)}")
    print(f"Test:  {'🟥' * len(X_test)}")

    print("✅ All rows accounted for!")

if __name__ == "__main__":
    test_split_data()

