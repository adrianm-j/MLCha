# Challenge

1. Run phrase_mach_dataset.py in order to generate labels by matching the words from taxonomy labels.
2. Run preprocessing_dataset.py in order to process the dataset: lowercase, keep only letters, remove stop words (spacy stop words), remove labes if there is any duplicate in the cell.
3. Run train_classifier_bert.py in order to train a pretrained BERT model using the generated dataset.
4. Run run_model.py in order to run the model on the labels that don't have a label on the label column.

   For more details, please check the code comments.
