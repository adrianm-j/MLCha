import pandas as pd
import re
import spacy

def preprocess_text(text):
    """Convert to lowercase, keep only letters and spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_duplicates(labels):
    """Split labels by comma, remove duplicates, and rejoin."""
    label_list = [label.strip() for label in labels.split(",")]
    unique_labels = list(set(label_list))  # Remove duplicates
    return ", ".join(unique_labels)

def remove_stopwords(text):
    """Remove stop words from the input text."""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(filtered_tokens)

# Load the data from the XLSX file
file_path = "ml_insurance_challenge_phrase_mach.xlsx"
data = pd.read_excel(file_path)

# Preprocess columns
data['description'] = data['description'].fillna('').apply(preprocess_text)
data['business_tags'] = data['business_tags'].fillna('').apply(preprocess_text)
data['sector'] = data['sector'].fillna('').apply(preprocess_text)
data['category'] = data['category'].fillna('').apply(preprocess_text)
data['niche'] = data['niche'].fillna('').apply(preprocess_text)

# The 'labels' column remains unchanged
data['labels'] = data['labels'].fillna('').apply(remove_duplicates)

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Preprocess columns
data['description'] = data['description'].apply(remove_stopwords)
data['business_tags'] = data['business_tags'].apply(remove_stopwords)
data['sector'] = data['sector'].apply(remove_stopwords)
data['category'] = data['category'].apply(remove_stopwords)
data['niche'] = data['niche'].apply(remove_stopwords)

# Split the labels column into individual labels
data['labels'] = data['labels'].apply(lambda x: [label.strip() for label in x.split(',')])

# Rename a column
data = data.rename(columns={'labels': 'insurance_label'})

# Expand the labels into separate rows for better readability
formatted_data = data.explode('insurance_label')

# Save the data into a CSV file
formatted_data.to_csv('formatted_dataset.csv', index=False)

print("Formatted dataset has been saved as 'formatted_dataset.csv'.")

