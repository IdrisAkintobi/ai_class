import pickle
with open("./data/word_to_vector.pkl", "rb") as pk:
    word_to_vector = pickle.load(pk)

"""
Purpose: Measures similarity between two vectors vec_a and vec_b.
numerator: Dot product of the vectors.
denominator: Product of magnitudes of the two vectors.

Returns a value between -1 and 1, where:
1 means highly similar
0 means orthogonal (no similarity)
-1 means opposite directions
"""
def cosine_similarity(vec_a, vec_b):
    numerator = sum([vec_a[i] * vec_b[i] for i in range(len(vec_a))])
    denominator = (sum([vec_a[i] ** 2 for i in range(len(vec_a))]) ** 0.5 * 
                   sum([vec_b[i] ** 2 for i in range(len(vec_b))]) ** 0.5)
    return numerator / denominator


"""
Purpose: Given a word, finds the top top_k most similar words based on cosine similarity.
Iterates over all words in the vocabulary.
Computes cosine similarity with the input word.
Sorts them in descending order (-cosine_similarity(...)).

Returns the top k most similar words.
"""
def similar_words(word="tree", top_k=10):
    return sorted(  
        word_to_vector.keys(), 
        key=lambda x: -cosine_similarity(word_to_vector[x], word_to_vector[word])
    )[:top_k + 1][1:] # Remove the first word, which is the word that is passed

def getSimilarity(word_a, word_b):
    return cosine_similarity(word_to_vector[word_a], word_to_vector[word_b])