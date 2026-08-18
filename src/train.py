import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_data = pd.read_csv("data/train.csv")

X = train_data.drop("target", axis=1)
y = train_data["target"]

model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["train"]["random_state"]
)

model.fit(X, y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training completed.")