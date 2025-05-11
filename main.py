from markov_chain import complete_prompt
from cosine_similarity import getSimilarity, similar_words
from naive_bayes_classification import get_sentiment


# Test MarkovChain
sample_prompt = "He was"
for i in range(5):
    print(complete_prompt(sample_prompt))


# Test Cos similarity
words_tuple = [("plant", "grow"), ("minute", "plant"), ("tree", "tree")]
for a, b in words_tuple:
    print(f"The similarity between {a} and {b} is: ",getSimilarity(a, b))
    print(f"The top 5 similar words to {a} are: ", similar_words(a, 4))

# Test Naive Bayes classification
comments = ["100 percent", "I hate this", "This is lovely"]
for comment in comments:
    print(f"The comment {comment} is {get_sentiment(comment)}")