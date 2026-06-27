from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["dataset"]

    x, y = make_classification(
        n_samples=params["n_samples"],
        n_features=params["n_features"],
        n_informative=params["n_informative"],
        n_redundant=params["n_redundant"],
        n_classes=params["n_classes"],
        random_state=params["random_state"],
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "dataset.npz",
        x_train=x_train.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
    )


if __name__ == "__main__":
    main()
