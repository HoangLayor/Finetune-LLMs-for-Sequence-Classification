*Finetune Flan-T5 for email classification*
# Prepare data
- Xử lý email và đưa vào file csv
```python
import pandas as pd
import email
import csv
from email import policy
import os
import re

# Sắp xếp file theo thứ tự
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def pandas_escape(text):
    df = pd.DataFrame([text])
    return df.to_csv(index=False, header=False, quoting=csv.QUOTE_ALL, escapechar='\\').strip()

def email_to_string(email_content):
    msg = email.message_from_string(email_content, policy=policy.default)

    headers = ['From', 'To', 'Subject', 'Date']
    email_data = ""
    for header in headers:
        if msg.get(header, '') == '':
            continue
        email_data += f"{header}: {msg.get(header, '')}\n"

    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = msg.get_payload(decode=True).decode()

    body = body.rstrip('\n')
    full_email = f"{email_data}Body: {body}"
    return pandas_escape(full_email)

def custom_data_to_csv(label, folder_path, csv_file):
    for file_name in sorted(os.listdir(folder_path), key=natural_sort_key):
        file_path = os.path.join(folder_path, file_name)
        email_str = ''
        with open(file_path, 'r') as f:
            email_str = f.read()
            email_str = email_to_string(email_str)
        with open(csv_file, 'a') as f:
            f.write(f"{file_name},{label},{email_str}\n")
```
```python
ROOT_DIR = "/teamspace/studios/this_studio"

spam_data_dir = os.path.join(ROOT_DIR, "data/TrainData/spam")
notspam_data_dir = os.path.join(ROOT_DIR, "data/TrainData/notspam")
test_data_dir = os.path.join(ROOT_DIR, "data/TestData_nolabel")

train_csv_file = os.path.join(ROOT_DIR, "BaiThi2/train_data.csv")
test_csv_file = os.path.join(ROOT_DIR, "BaiThi2/test_data.csv")
```
```python
# from custom_data import custom_data_to_csv

with open(train_csv_file, 'w') as csv_f:
    csv_f.write("filename,label,text\n")

custom_data_to_csv("spam", spam_data_dir, train_csv_file)
custom_data_to_csv("notspam", notspam_data_dir, train_csv_file)
df = pd.read_csv(train_csv_file)
print(df.head())
print(df.info())
df = df.drop(columns=['filename'])
df
```
# Load dataset
```python
import os
import numpy as np
import pandas as pd
from datasets import Dataset

label2id = {"notspam": 0, "spam": 1}
id2label = {id: label for label, id in label2id.items()}

dataset_email = pd.read_csv(train_csv_file)
dataset_email = dataset_email.drop(columns=["filename"])

def load_dataset(model_type: str = "") -> Dataset:
    """Load dataset."""

    dataset_email["label"] = dataset_email["label"].astype(str)
    if model_type == "AutoModelForSequenceClassification":
        # Convert labels to integers
        dataset_email["label"] = dataset_email["label"].map(
            label2id
        )

    dataset_email["text"] = dataset_email["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_email)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset
```
```python
# from data_loader import load_dataset

train_dataset = load_dataset("AutoModelForSequenceClassification")
train_dataset
```
```python
# Initialize base model and tokenizer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

label2id = {"notspam": 0, "spam": 1}
id2label = {id: label for label, id in label2id.items()}

MODEL_ID = "google/flan-t5-base"
REPOSITORY_ID = f"{MODEL_ID.split('/')[1]}-email-classification"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=len(label2id))
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```
```python
import evaluate
import nltk
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

metric = evaluate.load("accuracy")

def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred) -> dict:
   """Compute metrics for evaluation"""
   logits, labels = eval_pred
   if isinstance(
      logits, tuple
   ):  # if the model also returns hidden_states or attentions
      logits = logits[0]
   predictions = np.argmax(logits, axis=-1)
   precision, recall, f1, _ = precision_recall_fscore_support(
      labels, predictions, average="binary"
   )
   return {"precision": precision, "recall": recall, "f1": f1}
```
```python
tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
```
# Training
```python
training_args = TrainingArguments(
    num_train_epochs=5,
    output_dir=REPOSITORY_ID,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=False,  # Overflows with fp16
    learning_rate=3e-4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,
)
```
- Save model
```python
trainer.train()
```
```python
tokenizer.save_pretrained(REPOSITORY_ID)
print(trainer.evaluate())
```
# Evaluate
- Load model
```python
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
model = AutoModelForSequenceClassification.from_pretrained(REPOSITORY_ID)
model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
```
- Classify & Eval
```python
from time import time
from typing import List, Tuple

import torch
from loguru import logger
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def classify(texts_to_classify: List[str]) -> List[Tuple[str, float]]:
    """Classify a list of texts using the model."""
    # Tokenize all texts in the batch
    start = time()
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    logger.debug(
        f"Classification of {len(texts_to_classify)} examples took {time() - start} seconds"
    ) # logger

    # Process the outputs to get the probability distribution
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top class and the corresponding probability (certainty) for each input text
    confidences, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = (
        predicted_classes.cpu().numpy()
    )  # Move to CPU for numpy conversion if needed
    confidences = confidences.cpu().numpy()  # Same here

    # Map predicted class IDs to labels
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]

    # Zip together the predicted labels and confidences and convert to a list of tuples
    return list(zip(predicted_labels, confidences))

def eval():
    """Evaluate the model on the test dataset."""
    predictions_list, labels_list = [], []

    batch_size = 16  # Adjust batch size based GPU capacity
    num_batches = len(train_dataset["test"]) // batch_size + (
        0 if len(train_dataset["test"]) % batch_size == 0 else 1
    )
    progress_bar = tqdm(total=num_batches, desc="Evaluating")

    for i in range(0, len(train_dataset["test"]), batch_size):
        batch_texts = train_dataset["test"]["text"][i : i + batch_size]
        batch_labels = train_dataset["test"]["label"][i : i + batch_size]

        batch_predictions = classify(batch_texts)

        predictions_list.extend(batch_predictions)
        labels_list.extend([id2label[label_id] for label_id in batch_labels])

        progress_bar.update(1)

    progress_bar.close()
    report = classification_report(labels_list, [pair[0] for pair in predictions_list])
    print(report)

eval()
```
```python
predictions_list, labels_list = [], []

texts = train_dataset['test']['text']
labels = train_dataset['test']['label']

predictions = classify(texts)

predictions_list.extend(predictions)
labels_list.extend([id2label[label_id] for label_id in labels])
for id, prediction in enumerate(predictions):
    print(f"Actual Label: {labels_list[id]}\n>>> Prediction: {predictions_list[id]}")
```
