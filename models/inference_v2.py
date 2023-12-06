import pandas as pd
import numpy as np
import torch
import csv
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_test_ids(path):
    file = open(path,'r')
    lines = file.readlines()
    for rowidx in range(len(lines)):
        index = lines[rowidx].index(',')
        lines[rowidx] = lines[rowidx][:index]
    return lines    

def create_csv_submission(ids, y_pred, name):
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, labels):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.long)
        }

user = sys.argv[1]
if user == 'simon':
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_path = '../twitter-datasets/processed_test.csv'

test_processed = pd.read_csv(test_path)
test_tweets = test_processed['text'].values
test_ids = test_processed["ids"].values

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 128
BATCH_SIZE = 32

test_set = TweetDataset(test_tweets, tokenizer, MAX_LEN, test_ids)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained('/Users/simonli/Desktop/epfl/cs433/project2/sentiMentaL_tweets/model/distilbert_2023_12_06_00:24:04_0.8547515305278253%', num_labels=2)
model.to(device)

### Prediction ###
predictions = []

model.eval()
val_bar = tqdm(test_loader, desc=f"Testing")
for batch in val_bar:
    ids = batch['ids'].to(device)
    mask = batch['mask'].to(device)

    with torch.no_grad():
        outputs = model(ids, mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(probabilities, 1)
        predictions.extend(predicted.cpu().numpy())

#-----------------------------------------------------------------------------------------------------#
### CREATE CSV SUBMISSION ###

predictions = np.array(predictions)

ids_test = get_test_ids('../twitter-datasets/test_data.txt')
y_pred = []
y_pred = predictions
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
create_csv_submission(test_ids, y_pred, "../submissions/submission_distilbert_12_06.csv")