import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score

test_data_path = 'twitter-datasets/processed_test_data.txt'
train_pos_path = 'twitter-datasets/processed_train_pos_full.txt'
train_neg_path = 'twitter-datasets/processed_train_neg_full.txt'

# Load the test set tweets
with open(test_data_path, 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()

# Load positive training tweets and assign labels
with open(train_pos_path, 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

pos_labels = np.ones(len(pos_tweets), dtype=int)  # Assign label 1 for positive tweets

# Load negative training tweets and assign labels
with open(train_neg_path, 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

neg_labels = -1 * np.ones(len(neg_tweets), dtype=int)  # Assign label -1 for negative tweets

# Combine positive and negative tweets and labels
train_tweets = pos_tweets + neg_tweets
labels = np.concatenate((pos_labels, neg_labels), axis=0)

# Assuming you have preprocessed data and labels
X = np.array(train_tweets)  # Replace with your preprocessed sentences
y = np.array(labels)  # Replace with your sentiment labels

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize input data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_inputs_train = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
tokenized_inputs_test = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='pt')

# Create PyTorch DataLoader
train_dataset = TensorDataset(tokenized_inputs_train['input_ids'],
                              tokenized_inputs_train['attention_mask'],
                              torch.tensor(y_train))

test_dataset = TensorDataset(tokenized_inputs_test['input_ids'],
                             tokenized_inputs_test['attention_mask'],
                             torch.tensor(y_test))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Create optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 3  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs, attention_mask, labels = batch
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, attention_mask, labels = batch
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy * 100:.2f}%')

# Evaluate the final model
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        inputs, attention_mask, labels = batch
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Final Accuracy: {accuracy * 100:.2f}%')
