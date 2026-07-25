from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import DVCClassifier


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    torch.manual_seed(train_params["random_state"])
    np.random.seed(train_params["random_state"])

    data = np.load("data/raw/dataset.npz")

    x_train = torch.tensor(data["x_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)

    dataset = TensorDataset(x_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
    )

    model = DVCClassifier(
        input_dim=x_train.shape[1],
        hidden_dim=train_params["hidden_dim"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["learning_rate"],
    )

    model.train()

    for epoch in range(train_params["epochs"]):
        total_loss = 0.0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"epoch={epoch + 1}, loss={avg_loss:.4f}")

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "hidden_dim": train_params["hidden_dim"],
        },
        output_dir / "model.pt",
    )


if __name__ == "__main__":
    main()
