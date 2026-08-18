import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_data = X_train.copy()
train_data["target"] = y_train

test_data = X_test.copy()
test_data["target"] = y_test

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

print("Preprocessing completed.")