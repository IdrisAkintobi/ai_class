import pickle

try:
    with open("./data/word_to_vector.pkl", "rb") as pk:
        word_to_vector = pickle.load(pk)
except FileNotFoundError:
    print("Error: Could not find the word vector file. Please check the file path.")
    word_to_vector = {}


def cosine_similarity(vec_a, vec_b) -> float:
    """
    Purpose: Measures similarity between two vectors vec_a and vec_b.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        A value between -1 and 1, where:
        1 means highly similar
        0 means orthogonal (no similarity)
        -1 means opposite directions
    """
    # Validate input dimensions
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimensions")

    # Calculate dot product
    numerator = sum([vec_a[i] * vec_b[i] for i in range(len(vec_a))])

    # Calculate magnitudes
    mag_a = sum([vec_a[i] ** 2 for i in range(len(vec_a))]) ** 0.5
    mag_b = sum([vec_b[i] ** 2 for i in range(len(vec_b))]) ** 0.5

    # Handle zero magnitude case
    if mag_a == 0 or mag_b == 0:
        return 0.0  # Vectors with zero magnitude have no direction, so similarity is 0

    return numerator / (mag_a * mag_b)


def similar_words(word="tree", top_k=10):
    """
    Purpose: Given a word, finds the top top_k most similar words based on cosine similarity.

    Args:
        word: The target word to find similarities for
        top_k: Number of similar words to return

    Returns:
        List of the top k most similar words
    """
    # Check if word exists in our vector dictionary
    if word not in word_to_vector:
        raise KeyError(f"Word '{word}' not found in the word vector dictionary")

    similarities = []
    for candidate_word, vector in word_to_vector.items():
        if candidate_word != word:  # Skip the input word itself
            try:
                similarity = cosine_similarity(vector, word_to_vector[word])
                similarities.append((candidate_word, similarity))
            except Exception as e:
                print(f"Error computing similarity for {candidate_word}: {e}")

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return just the words (not the similarity scores)
    return [word for word, _ in similarities[:top_k]]


def get_similarity(word_a, word_b):
    """
    Purpose: Calculate the similarity between two specific words

    Args:
        word_a: First word
        word_b: Second word

    Returns:
        Cosine similarity between the word vectors
    """
    # Check if both words exist in our dictionary
    if word_a not in word_to_vector:
        raise KeyError(f"Word '{word_a}' not found in the word vector dictionary")
    if word_b not in word_to_vector:
        raise KeyError(f"Word '{word_b}' not found in the word vector dictionary")

    return cosine_similarity(word_to_vector[word_a], word_to_vector[word_b])
