import pickle
from string import punctuation
from collections import Counter
from collections import defaultdict

with open("./data/comments_with_labels.pkl", "rb") as f:
    post_comments_with_labels = pickle.load(f)

print(post_comments_with_labels)

class NaiveBayesClassifier:
    def __init__(self, samples):
        self.mapping = {"pos": [], "neg": []}
        self.neg_mapping = defaultdict(lambda: 0)
        self.sample_count = len(samples)
        for text, label in samples:
            self.mapping[label] += self.tokenize(text)
        self.pos_counter = Counter(self.mapping["pos"])
        self.neg_counter = Counter(self.mapping["neg"])

    @staticmethod
    def tokenize(text):
        return (
            text.lower().translate(str.maketrans("", "", punctuation + "1234567890"))
            .replace("\n", " ")
            .split(" ")
        )

    def classify(self, text):
        tokens = self.tokenize(text)
        pos = []
        neg = []

        for token in tokens:
            pos.append(self.pos_counter[token]/ self.sample_count)
            neg.append(self.neg_counter[token]/self.sample_count)

        total_pos = sum(pos)
        total_neg = sum(neg)
        if(total_neg > total_pos):
            return "neg"
        elif (total_neg < total_pos):
            return "pos"
        return "neutral"


def get_sentiment(text):
    cl = NaiveBayesClassifier(post_comments_with_labels)
    return cl.classify(text)

