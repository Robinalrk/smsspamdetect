import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

data=pd.read_csv('combined_file.csv')

#checking null values
print(data.isnull().sum())

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [PorterStemmer().stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Apply preprocessing
data['Message'] = data['Message'].apply(preprocess_text)
print(data.head())
processed_file_path = "processed_dataset.csv"
data.to_csv(processed_file_path, index=False)
print(f"Processed data saved to {processed_file_path}")