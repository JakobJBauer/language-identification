import re

def preprocess_text(text):
    """Clean and preprocess text."""
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates
    text = re.sub(r'\[\d+\]', '', text)      # Remove references
    text = re.sub(r'\[citation needed\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()
    return text
