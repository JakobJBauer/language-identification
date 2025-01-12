from sklearn.metrics.pairwise import cosine_similarity
from scripts.preprocessing import preprocess_text
from scripts.train import train_language_model
from scripts.utils import model_to_vector, sharpen_distribution


def detect_language(input_text, language_models, n, alpha=1):
    # Preprocess and create the input n-gram model
    input_model = train_language_model([preprocess_text(input_text)], n)

    # Unified vocabulary of all n-grams
    all_ngrams = set(ngram for model in language_models.values() for ngram in model.keys()).union(input_model.keys())

    input_vector, input_vector_norm = model_to_vector(input_model, all_ngrams, alpha)

    # Compute similarities with each language model
    similarities = {}
    for lang, model in language_models.items():
        model_vector, model_vector_norm = model_to_vector(model, all_ngrams, alpha)

        # Calculate cosine similarity
        if input_vector_norm == 0 or model_vector_norm == 0:
            similarities[lang] = 0.0
        else:
            similarities[lang] = cosine_similarity([input_vector], [model_vector])[0][0]

    # Normalize similarities to sum to 1
    # total_similarity = sum(similarities.values())
    # if total_similarity > 0:
    #     probabilities = {lang: sim / total_similarity for lang, sim in similarities.items()}
    # else:
    #     probabilities = {lang: 1 / len(language_models) for lang in language_models}  # Uniform probabilities

    return sharpen_distribution(sharpen_distribution(similarities, temperature=0.01),temperature=0.01)
