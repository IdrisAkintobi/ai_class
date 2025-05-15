import spacy

# Load model
nlp = spacy.load("en_core_web_md")


def process_documents(texts):
    """Process a list of texts using the spaCy pipeline."""
    # Print model pipelines
    print("Spacy pipeline components:", nlp.pipe_names)
    return [nlp(text) for text in texts]


def compute_similarities(doc1, doc2):
    """Compute and print similarities between documents."""
    print(f"Similarity between doc1 and doc2: {doc1.similarity(doc2):.4f}")
    print(f"Similarity between doc1 and itself: {doc1.similarity(doc1):.4f}")
