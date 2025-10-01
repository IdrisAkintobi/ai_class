from basics.cosine_similarity import get_similarity, similar_words
from basics.markov_chain import complete_prompt
from basics.naive_bayes_classification import get_sentiment
from shared.print_header import print_header
from spacy_utils import compute_similarities, process_documents

# Test MarkovChain
print_header('MarkovChain Test')
sample_prompt = 'He was'
for _i in range(5):
    print(complete_prompt(sample_prompt))


# Test Cos similarity
print_header('Cosine Similarity Test -- words similarity score')
words_tuple = [('plant', 'grow'), ('minute', 'plant'), ('tree', 'tree')]
for a, b in words_tuple:
    print(f'The similarity between {a} and {b} is: ', get_similarity(a, b))

print_header('Cosine Similarity Test -- similar words')
num_of_words = 3
for a, _b in words_tuple:
    print(
        f'The top {num_of_words} similar words to {a} are: ',
        similar_words(a, num_of_words),
    )

# Test Naive Bayes classification
print_header('Naive Bayes Classification Test')
comments = ['100 percent', 'I hate this', 'This is lovely']
for comment in comments:
    print(f'The comment {comment} is {get_sentiment(comment)}')


print_header('Texts similarity test using spacy')
texts = [
    'Apple is looking at buying U.K. startup for $1 billion',
    'Microsoft is looking at buying NG startup for $4 billion',
]
doc1, doc2 = process_documents(texts)
compute_similarities(doc1, doc2)
