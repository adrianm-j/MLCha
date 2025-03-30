import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.optim as optim
import json

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
    
def train_model(model, dataloader, criterion, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Load dataset
df = pd.read_csv("formatted_dataset.csv") 

# Remove rows where 'labels' column is empty
df = df[df['insurance_label'].notna()] 

# Extract features and labels
features = df[['description', 'business_tags', 'sector', 'category', 'niche']].fillna('')

# Combine features into a single text column
df['text'] = features.apply(lambda x: ' '.join(x), axis=1)

df['insurance_label'] = df['insurance_label'].apply(lambda x: x.split(','))  # Convert string to list

# Encode labels
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(df['insurance_label'])

num_labels = len(mlb.classes_)
# print(mlb.classes_)
# print(num_labels)
print(f"Number of classes found {len(mlb.classes_)}")

data = pd.read_csv("labels_list.csv")
# print(data.iloc[:, 0].tolist())

not_common_elements = list(set(mlb.classes_).symmetric_difference(set(data.iloc[:, 0].tolist())))
print(not_common_elements)


# Create a label mapping
dict_label_map = {idx: label for idx, label in enumerate(mlb.classes_)}

# Save to labels to JSON to be used when infericing 
with open("label_map.json", "w") as f:
    json.dump(dict_label_map, f, indent=4)

print("Label map saved to label_map.json")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create dataset and DataLoader
dataset = TextDataset(df["text"].tolist(), labels_encoded)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
# Check if the training will run on CUDA (GPU) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Train on {device}")

train_model(model, dataloader, criterion, optimizer, epochs=3)

torch.save(model.state_dict(), "bert_model.pth")