import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import json

NUM_LABELS = 183

def get_top_labels_with_threshold(text, threshold=0.01, n=5):
    probabilities = predict(text)  # Get model output

    # Or use threshold if you want to
    # threshold_probabilities = torch.nonzero(probabilities > threshold).squeeze().tolist()
    # threshold_probabilities_list = [x[1] for x in threshold_probabilities]

    """Selects the top N labels with scores exceeding a threshold."""
    top_indices = torch.topk(probabilities, n).indices.tolist()
    # print(top_indices)
    top_labels = [label_map[idx] for idx in top_indices[0]]
    # print(top_labels)

    return top_labels    

def predict(text):
    """Function to run inference on a given text"""
    #Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Run the model and get logits
    with torch.no_grad():
        outputs = model(**inputs)

    #Apply sigmoid for multi-label classification or softmax for multi-class classification
    logits = outputs.logits
    #print(logits)
    probabilities = torch.sigmoid(logits) if NUM_LABELS > 1 else torch.softmax(logits, dim=1)

    return probabilities


#Load the tokenizer (use the same model name as used during training)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Define the model architecture (should match the training phase)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)

#Load the saved model weights
model.load_state_dict(torch.load("bert_model.pth", map_location=torch.device("cpu")))

#Set the model to evaluation mode
model.eval()



#Load labels from JSON
with open("label_map.json", "r") as f:
    dict_label_map = json.load(f)

# print(dict_label_map)

#Reverse mapping for lookup
label_map = {int(idx): label for idx, label in dict_label_map.items()}

df = pd.read_excel("ml_insurance_challenge_phrase_mach.xlsx")
features = df[['description', 'business_tags', 'sector', 'category', 'niche']].fillna('')
df['text'] = features.apply(lambda x: ' '.join(x), axis=1)

if 'insurance_label' in df.columns:
    #Iterate through each row and check for empty values in the labels column
    for index, row in df.iterrows():
        if pd.isna(row['labels']):  #Check if the value is NaN (empty)
            print(index)
            # print(df['text'][index])
            predictions = get_top_labels_with_threshold(df['text'][index])
            print(predictions)
else:
    print("The 'labels' column is not present in the Excel file.")
