import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = pd.read_csv("data/test.csv")

X = test_data.drop("target", axis=1)
y = test_data["target"]

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X)

metrics = {
    "accuracy": accuracy_score(y, predictions),
    "precision": precision_score(y, predictions, average="weighted"),
    "recall": recall_score(y, predictions, average="weighted"),
    "f1_score": f1_score(y, predictions, average="weighted")
}

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)