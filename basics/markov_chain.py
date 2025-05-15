import random
from string import punctuation
from collections import defaultdict

with open("./data/text.txt", "r") as data:
    data = data.read()


class MarkovChain:
    def __init__(self):
        self.graph = defaultdict(list)

    def _tokenize(self, text):
        return (
            text.translate(str.maketrans("", "", punctuation + "1234567890"))
            .replace("\n", " ")
            .split(" ")
        )

    def train(self, text):
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            if (len(tokens) - 1) == i:
                break
            self.graph[token].append(tokens[i + 1])

    def generate(self, prompt, length):
        # get the last token from the prompt
        current = self._tokenize(prompt)[-1]  # Get the last word of the prompt
        output = prompt  # Start with the prompt as the output string
        for i in range(length):
            options = self.graph.get(current, [])  # Get possible next words
            if not options:  # If no options are available, skip
                continue
            current = random.choice(options)  # Pick a random next word
            output += " " + current  # Append the new word to the output
        return output  # Return the generated output string


# Create class instance and train on the available data
chain = MarkovChain()
chain.train(data)


# Provide a prompt that will be completed with the trained data
def complete_prompt(prompt, length=10):
    return chain.generate(prompt, length)
