import os

import pandas as pd
from datasets import Dataset

label2id = {"notspam": 0, "spam": 1}
id2label = {id: label for label, id in label2id.items()}
train_csv_file = "/teamspace/studios/this_studio/BaiThi2/train_data.csv"

def load_dataset(model_type: str = "") -> Dataset:
    """Load dataset."""
    dataset_email = pd.read_csv(train_csv_file)
    dataset_email = dataset_email.drop(columns=['filename'])

    dataset_email["label"] = dataset_email["label"].astype(str)
    if model_type == "AutoModelForSequenceClassification":
        # Convert labels to integers
        dataset_email["label"] = dataset_email["label"].map(
            label2id
        )

    dataset_email["text"] = dataset_email["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_email)
    dataset = dataset.train_test_split(test_size=0.99, seed=42)

    return dataset


if __name__ == "__main__":
    print(load_dataset())