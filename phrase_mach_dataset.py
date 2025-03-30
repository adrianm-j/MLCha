import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
from openpyxl import load_workbook
import re
import csv

def keep_letters(cell):
    if isinstance(cell, str):  # Check if the cell is a string
        return ''.join(re.findall(r'[a-zA-Z\s]', cell))
    return cell

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# The labels to match
lables_taxonomy_file = 'insurance_taxonomy.xlsx'

labels_raw_data = pd.read_excel(lables_taxonomy_file)

# Select specific columns
selected_labels_data = labels_raw_data.iloc[:, [0]] 

# new_labels_format = list()

for x in range(0, len(selected_labels_data)):
# for x in range(10, 50):
    # print(type(selected_labels_data))
    # print(str(selected_labels_data["label"][x]).split( v))
    
    #Take the elements from the label column
    label_phrases = str(selected_labels_data["label"][x]).split()

    #Lowercase all the elements in the list
    label_phrases = [element.lower() for element in label_phrases]
    

    #Keep only the letters and remove others signs
    label_phrases = ["".join(char for char in s if char.isalpha()) for s in label_phrases]

    #Remove the stop words
    words_to_remove = ["the", "a", "an", "in", "of", "and"]
    label_phrases = [word for word in label_phrases if word not in words_to_remove]

    #Remove empty and none elements from the list
    label_phrases = [item for item in label_phrases if item]

    print("Search for {}".format(label_phrases))

    # new_labels_format.append(" ".join(label_phrases))

    companies_data_file = 'ml_insurance_challenge.csv'
    companies_raw_data = pd.read_csv(companies_data_file)
    selected_companies_data = companies_raw_data.iloc[:, [0, 1, 2, 3, 4]] 

    companies_data = selected_companies_data.to_dict(orient='records')

    df = pd.DataFrame(companies_data)

    #print(df['description'])

    #Fill missing values with empty strings
    df['description'] = df['description'].fillna("")
    df['business_tags'] = df['business_tags'].fillna("")
    df['sector'] = df['sector'].fillna("")
    df['category'] = df['category'].fillna("")
    df['niche'] = df['niche'].fillna("")

    #Combine all under one column
    df['text_unfiltered'] = df['description'] + " " + df['business_tags']  + " " + df['sector'] + " " + df['category'] + " " + df['niche']

    #Filter the text and keep only the letters and spaces
    df['text'] = df['text_unfiltered'].apply(keep_letters)

    #Create the PhraseMatcher and add patterns
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(phrase) for phrase in label_phrases]
    matcher.add("LabelMatcher", patterns)

    patterns = [nlp.make_doc(term) for term in label_phrases]
    matcher.add("TechnologyTerms", patterns)

    #Function to run PhraseMatcher on a text
    def find_matches(text):
        # text = ["".join(char for char in s if char.isalpha()) for s in text]
        
        doc = nlp(text.lower())
        matches = matcher(doc)
        # print(text)
        
        #Check if the number of unique words matched are the same as in the label_phrase
        # This solution can be improved by counting the less occured word and remove it
        # to get more matches, but with less accuracy
        if len(label_phrases) == len(set([doc[start:end].text for _, start, end in matches])):
            print(set([doc[start:end].text for _, start, end in matches]))
            return " ".join(label_phrases)


    #Apply the matcher to each row in the DataFrame
    df['matches'] = df['text'].apply(find_matches)

    # Path to the Excel file
    excel_file = 'ml_insurance_challenge_phrase_mach.xlsx'

    #Name of the target Excel sheet
    sheet_name = 'ml_insurance_challenge'

    #Load the workbook and sheet
    workbook = load_workbook(excel_file)
    sheet = workbook[sheet_name]

    #Define the target column in Excel
    target_column = 'F'
    start_row = 2  # Starting row for data insertion

    #Iterate through the DataFrame and Excel rows
    for i, value in enumerate(df['matches'], start=start_row):
        cell = sheet[f"{target_column}{i}"]
        if value is not None: # If value is not None
            if cell.value:  # If cell is not empty
                cell.value = f"{cell.value}, {value}"  # Append new text
            else:  # If cell is empty
                cell.value = value  # Write new value

    #Save changes to the Excel file
    workbook.save(excel_file)
    print("Data saved successfully!")






