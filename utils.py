import os
import zipfile

def extract_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data')

def read_documents(data_path):
    documents = {}
    for file in os.listdir(data_path):
        if file.endswith('.txt'):
            with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                documents[file] = f.read().strip()
    return documents