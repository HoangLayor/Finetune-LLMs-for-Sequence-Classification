import pandas as pd
import email
import csv
from email import policy
import os

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

def custom_data_to_csv(train_data_dir, csv_file):
    for folder_name in os.listdir(train_data_dir):
        label = folder_name # notspam, spam
        folder_path = os.path.join(train_data_dir, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            email_str = ''
            with open(file_path, 'r') as f:
                email_str = f.read()
                email_str = email_to_string(email_str)
            with open(csv_file, 'a') as f:
                f.write(f"{label},{email_str}\n")

if __name__ == "__main__":
    ROOT_DIR = "/teamspace/studios/this_studio"

    train_data_dir = os.path.join(ROOT_DIR, "data/TrainData")
    test_data_dir = os.path.join(ROOT_DIR, "data/TestData_nolabel")
    csv_file = os.path.join(ROOT_DIR, "train_data.csv")

    with open(csv_file, 'w') as csv_f:
        csv_f.write("label,text\n")

    custom_data_to_csv(train_data_dir, csv_file)