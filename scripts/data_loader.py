import os
import json

def load_wikimedia_texts(json_folder):
    """Load text from WikiExtractor JSON output."""
    texts = []
    for root, _, files in os.walk(json_folder):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line)
                    texts.append(article['text'])  # Extract plain text
    return texts
