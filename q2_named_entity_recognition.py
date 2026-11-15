"""
Home Assignment 4 - Part II, Q2: Named Entity Recognition with spaCy
CS5720 Neural Network and Deep Learning - Fall 2026
Student: Palash Palit

Setup (one time):
    pip install spacy
    python -m spacy download en_core_web_sm
"""
import spacy

# Load spaCy's small English pipeline, which includes a trained NER component
nlp = spacy.load("en_core_web_sm")

# Input sentence exactly as given in the assignment (including the leading space)
sentence = " LeBron James as the NBA player of the United States in Laker and won the NBA Final MVP in 2020."

# Run the pipeline: tokenization, tagging, parsing and named entity recognition
doc = nlp(sentence)

print("Sentence:", repr(sentence))
print()
print(f"{'Entity text':<22}{'Label':<10}{'Start':>6}{'End':>6}   Label meaning")
print("-" * 70)
for ent in doc.ents:
    # ent.start_char / ent.end_char are character positions in the original string
    # (end is exclusive, so sentence[start:end] gives back the entity text)
    print(f"{ent.text:<22}{ent.label_:<10}{ent.start_char:>6}{ent.end_char:>6}   {spacy.explain(ent.label_)}")
    assert sentence[ent.start_char:ent.end_char] == ent.text   # check the positions are correct

print()
print("Total entities found:", len(doc.ents))
