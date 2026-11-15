"""
Home Assignment 4 - Part II, Q1: NLP Preprocessing Pipeline
CS5720 Neural Network and Deep Learning - Fall 2026
Student: Palash Palit

Steps: 1) tokenize  2) remove English stopwords  3) apply Porter stemming
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the NLTK data the pipeline needs, only if it is not installed yet
for path, resource in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab"),
                       ("corpora/stopwords", "stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


def preprocess(sentence):
    """Tokenize, remove stopwords and stem a sentence; print the result of each step."""
    # Step 1: Tokenization - split the sentence into words and punctuation marks
    tokens = word_tokenize(sentence)
    print("1. Original Tokens:", tokens)

    # Step 2: Stopword removal - drop common words ("the", "in", "are", ...).
    # Comparison is lowercase so "And" and "and" are both removed.
    # Punctuation is also dropped here because it carries no meaning on its own.
    stop_words = set(stopwords.words("english"))
    filtered = [t for t in tokens if t.lower() not in stop_words and t.isalnum()]
    print("2. Tokens Without Stopwords:", filtered)

    # Step 3: Stemming - cut each word down to its root with the Porter stemmer
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in filtered]
    print("3. Stemmed Words:", stemmed)
    return tokens, filtered, stemmed


if __name__ == "__main__":
    sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
    print("Input sentence:", sentence)
    print()
    preprocess(sentence)
