import pickle
import numpy as np

def save_model(model, filename):
    """Save model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load model from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def keep_top_k_ngrams(ngram_dict, k=5000):
    # Sort by frequency (descending)
    sorted_ngrams = sorted(ngram_dict.items(), key=lambda x: x[1], reverse=True)
    # Keep top K
    top_k = dict(sorted_ngrams[:k])
    return top_k


def model_to_vector(input_model, all_ngrams, alpha=1e-8):
    """
    A minimal approach:
    - input_model: dictionary {ngram: freq}
    - all_ngrams: the *filtered* set of ngrams
    - alpha: small probability for unseen n-grams
    """
    import numpy as np

    total_count = sum(input_model.values())
    # Pre-allocate
    vec = np.full(len(all_ngrams), alpha, dtype=float)

    ngram_index = {ng: idx for idx, ng in enumerate(all_ngrams)}

    for ng, c in input_model.items():
        if ng in ngram_index:
            # Convert freq to probability but skip fancy add-alpha
            vec[ngram_index[ng]] = c / total_count

    # Now normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec, norm

def sharpen_distribution(probabilities, temperature=5):
    scores = np.array(list(probabilities.values()))

    exp_scores = np.exp(scores / temperature)
    exp_sum = np.sum(exp_scores)
    sharpened = exp_scores / exp_sum

    return {key: sharpened[i] for i, key in enumerate(probabilities.keys())}
