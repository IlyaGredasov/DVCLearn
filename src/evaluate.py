import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.model import DVCClassifier


def main() -> None:
    data = np.load("data/raw/dataset.npz")
    checkpoint = torch.load("models/model.pt", weights_only=True)

    x_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = data["y_test"]

    model = DVCClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        predictions = torch.argmax(logits, dim=1).numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
