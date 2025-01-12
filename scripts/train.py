from collections import Counter
import nltk
from nltk.util import ngrams

from scripts.utils import keep_top_k_ngrams


def train_language_model(texts, max_n, k=10000):
    """Train an n-gram model for a language."""
    ngram_counts = Counter()
    for text in texts:
        words = nltk.word_tokenize(text.lower())
        for n in range(1, max_n + 1):
            ngram_counts.update(ngrams(words, n))

    filtered_ngram_counts = keep_top_k_ngrams(ngram_counts, k)
    total_count = sum(filtered_ngram_counts.values())
    return {k: v / total_count for k, v in filtered_ngram_counts.items()}
