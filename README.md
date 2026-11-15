# Home Assignment 4 — Neural Networks and Deep Learning (CS5720)

**Student Name:** Palash Palit
**Course:** CS5720 Neural Network and Deep Learning — Fall 2026
**University:** University of Central Missouri, Department of Computer Science & Cybersecurity

## Overview

This repository contains Home Assignment 4, which covers sequence-to-sequence models, attention, Transformers and NLP. Part I has the written short answers, and Part II has three Python programs with their outputs and short answers.

| File | What it is |
|---|---|
| `q1_nlp_preprocessing.py` | Part II Q1: tokenization, stopword removal and stemming with NLTK |
| `q2_named_entity_recognition.py` | Part II Q2: named entity recognition with spaCy |
| `q3_scaled_dot_product_attention.py` | Part II Q3: scaled dot-product attention with NumPy |
| `q1_output.txt`, `q2_output.txt`, `q3_output.txt` | Output of each program from my run |
| `HomeAssignment4_Colab.ipynb` | Google Colab notebook that installs the libraries and runs all three programs |

## How to run

```bash
pip install nltk spacy numpy
python -m spacy download en_core_web_sm
python q1_nlp_preprocessing.py
python q2_named_entity_recognition.py
python q3_scaled_dot_product_attention.py
```

Or open `HomeAssignment4_Colab.ipynb` in Google Colab and click **Runtime → Run all**. Q1 downloads its NLTK data automatically the first time. Tested with Python 3.13, NLTK 3.10, spaCy 3.8 and NumPy 2.5.

# Part I. Short Answer

## Question 1. Sequence-to-Sequence Models and Attention

**(a) Purpose of the encoder and decoder in a Seq2Seq model for machine translation.**

The **encoder** reads the source sentence (for example English) one word at a time and turns it into hidden states that represent its meaning. In a basic Seq2Seq model, the encoder's final hidden state becomes the *context vector*, a summary of the whole sentence. The **decoder** takes this context vector and generates the target sentence (for example German) one word at a time. At each step it uses its own hidden state and the word it produced in the previous step to predict the next word, until it outputs the end-of-sentence token. Because the encoder and decoder are separate, the input and output sentences can have different lengths.

**(b) Why a fixed-length context vector performs poorly on long sentences.**

The whole source sentence must be squeezed into one fixed-size vector, no matter how long the sentence is. This is an information bottleneck: a 5-word sentence and a 50-word sentence get the same amount of space, so details from long sentences are lost. RNN encoders also tend to forget early words because of vanishing gradients, so the beginning of a long sentence is weakly represented in the final hidden state. As a result, translation quality drops sharply as sentences get longer.

**(c) How attention addresses this limitation, and what the decoder can access.**

With attention, the decoder no longer depends on a single context vector. It can access **all of the encoder's hidden states**, one for each source word. At every decoding step, it compares its current state with each encoder hidden state, turns these scores into weights with softmax, and builds a new context vector as the weighted sum of the encoder states. The decoder therefore gets a fresh, step-specific summary that focuses on the source words most relevant to the word it is producing now (for example, the source word "dog" when producing "Hund"). Information no longer has to pass through one bottleneck vector, so long sentences are handled much better, and the attention weights also show an alignment between source and target words.

## Question 2. Attention Mechanisms

**(a) Hard vs. soft attention, and why soft attention is easier to train.**

**Hard attention** selects exactly one encoder state (or image region) at each step, usually by sampling, and ignores the rest. **Soft attention** gives every encoder state a weight between 0 and 1 (all weights add up to 1) and uses the weighted average of all states.

Soft attention is easier to train because it is built only from smooth, differentiable operations: dot products, softmax and a weighted sum. Gradients flow through the attention weights, so the whole model can be trained end to end with ordinary backpropagation. Hard attention involves a discrete choice (argmax or sampling), which has no useful gradient, so it needs methods such as reinforcement learning (REINFORCE), which are noisy and slow.

**(b) Dot-product attention scores, eᵢ = sᵀhᵢ.**

```
s = [1, 0]

e1 = sᵀh1 = [1, 0]·[1, 0] = (1)(1) + (0)(0) = 1
e2 = sᵀh2 = [1, 0]·[0, 1] = (1)(0) + (0)(1) = 0
e3 = sᵀh3 = [1, 0]·[1, 1] = (1)(1) + (0)(1) = 1
```

**Attention scores: e = [1, 0, 1]**

**(c) Softmax attention weights and context vector (3 decimal places).**

```
exp(e1) = e^1 = 2.718
exp(e2) = e^0 = 1.000
exp(e3) = e^1 = 2.718
Sum      = 2.718 + 1.000 + 2.718 = 6.437

α1 = 2.718 / 6.437 = 0.422
α2 = 1.000 / 6.437 = 0.155
α3 = 2.718 / 6.437 = 0.422
```

**Attention weights: α = [0.422, 0.155, 0.422]** (these add up to 1; with unrounded values, 0.4223 + 0.1554 + 0.4223 = 1.000).

```
c = α1·h1 + α2·h2 + α3·h3
  = 0.4223·[1, 0] + 0.1554·[0, 1] + 0.4223·[1, 1]

First component  = 0.4223(1) + 0.1554(0) + 0.4223(1) = 0.845
Second component = 0.4223(0) + 0.1554(1) + 0.4223(1) = 0.578
```

**Context vector: c = [0.845, 0.578]**

The decoder state s = [1, 0] matches h₁ and h₃ (both have a 1 in the first position), so they get the largest weights, while h₂ = [0, 1] gets the smallest.

## Question 3. Self-Attention in Transformers

**(a) What self-attention is, and how it differs from encoder–decoder attention.**

Self-attention lets every token in a sequence look at every other token **in the same sequence** and build a new, context-aware representation of itself as a weighted combination of all the tokens. In traditional Seq2Seq attention, the attention goes **between two different sequences**: the decoder's hidden state (target side) attends to the encoder's hidden states (source side). In self-attention, the queries, keys and values all come from the same sentence, and all tokens are processed at the same time instead of step by step with an RNN.

**(b) Roles of Query (Q), Key (K) and Value (V).**

Each token's embedding is multiplied by three learned weight matrices to produce its Q, K and V vectors.

- **Query (Q):** what the current token is looking for. It is compared with the keys of all tokens.
- **Key (K):** what each token offers, used to match against queries. The dot product Q·K gives the attention score, meaning how relevant that token is to the current one.
- **Value (V):** the actual information each token passes on. After softmax turns the scores into weights, the output is the weighted sum of the value vectors.

A useful comparison is a search: the query is the search request, the keys are the titles that get matched, and the values are the content you retrieve.

**(c) How self-attention captures relationships between distant words, with an example.**

In self-attention, every token is connected directly to every other token in a single step, so the distance between two words does not matter: the first and last word of a sentence are just one attention step apart. An RNN has to pass information through every word in between, so it weakens over long distances.

Example: *"The **animal** didn't cross the street because **it** was too tired."* To understand "it," the model must link it to "animal," several words earlier. In self-attention, the query of "it" matches strongly with the key of "animal," so "animal" gets a high attention weight, and the representation of "it" absorbs the meaning of "animal." If the sentence said "because it was too wide," the attention would instead move to "street."

## Question 4. Multi-Head Attention

**(a) What multi-head attention is, and why multiple heads are used.**

Multi-head attention runs several self-attention operations (heads) in parallel. Each head has its own learned Q, K and V projection matrices, so each one looks at the sequence in a different representation subspace and learns a different attention pattern. A single head produces only one set of attention weights, which has to average all relationships together. With several heads, the model can capture several kinds of relationships at once, for example one head for grammar (subject and verb), one for coreference ("it" and "animal"), and one for neighboring words, which gives a richer representation.

**(b) Dimension of each head (embedding dimension 512, 8 heads).**

```
d_head = d_model / number of heads = 512 / 8 = 64
```

**Each attention head has dimension 64.** Concatenating all 8 heads gives 8 × 64 = 512 again, so the total cost is about the same as one 512-dimensional head.

**(c) How the head outputs are combined.**

The outputs of all heads are **concatenated** (joined side by side) for each token, giving a vector of size 8 × 64 = 512. This vector is then multiplied by a learned output projection matrix **W_O** (512 × 512), which mixes the information from the different heads into the final representation:

```
MultiHead(Q, K, V) = Concat(head1, head2, ..., head8) · W_O
```

## Question 5. Transformer Architecture

**(a) Why Transformers need positional encodings.**

Self-attention processes all tokens at the same time and treats them like an unordered set, so on its own it cannot tell word order: "dog bites man" and "man bites dog" would give the same attention results. Positional encodings are vectors added to each token's embedding that represent its position in the sequence (the original Transformer uses sine and cosine functions of different frequencies; other models learn them). They give the model the **order of the words and the distance between them**, so it can use word order in its understanding.

**(b) Masked self-attention in the decoder, and why future tokens must be hidden.**

Masked self-attention is self-attention in which every token can attend only to itself and the tokens before it. Before softmax, the scores for future positions are set to −∞, so their attention weights become 0.

The decoder is trained to predict the next token. During training, the whole target sentence is given to it at once (teacher forcing). If a token could attend to future tokens, it could just look at the answer it is supposed to predict, and it would not learn real prediction. At generation time the future tokens do not exist yet, so the masking also makes training match how the model is actually used.

**(c) RNNs vs. Transformers: sequential processing and parallel computation.**

An RNN processes tokens one after another, because each hidden state depends on the previous one. This cannot be parallelized across the sequence, so training is slow, and long-range information fades. A Transformer processes all tokens of a sequence in parallel with self-attention, and every token connects directly to every other token.

- **Advantage of Transformers:** training is much faster, because all positions are computed at once on GPUs, and they capture long-range dependencies better. This is what made very large models such as BERT and GPT possible.
- **Limitation in autoregressive generation:** at inference time the output is still produced one token at a time, because each new token depends on the tokens generated before it. Each step has to attend over the entire previous sequence, and self-attention's cost grows with the square of the sequence length, so generating long texts is slow and memory-heavy (in practice, the stored keys and values grow with every new token).

# Part II. Programming

## Q1: NLP Preprocessing Pipeline

**File:** `q1_nlp_preprocessing.py`

The function `preprocess(sentence)` uses NLTK in three steps:

1. **Tokenization:** `word_tokenize` splits the sentence into words and punctuation.
2. **Stopword removal:** removes every token in NLTK's English stopword list (compared in lowercase) and drops punctuation.
3. **Stemming:** the Porter stemmer cuts each remaining word down to its root.

**Output:**

```
Input sentence: NLP techniques are used in virtual assistants like Alexa and Siri.

1. Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
2. Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']
3. Stemmed Words: ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

"are," "in" and "and" were removed as stopwords, along with the period. The stemmer produced roots such as "techniqu" and "assist." A stem does not have to be a real word.

### Short Answer Questions

**1. Stemming vs. lemmatization, with the word "running."**

**Stemming** cuts word endings off using fixed rules, without checking a dictionary, so the result may not be a real word. **Lemmatization** uses a dictionary (vocabulary) and the word's part of speech to return its proper base form, the *lemma*, which is always a real word.

- "running": the Porter stemmer gives **"run"** (it removes "-ning"), and the lemmatizer treating it as a verb also gives **"run."**
- The difference shows on other words: stemming turns "studies" into "studi" (not a word), while lemmatization gives "study." Stemming cannot link irregular forms such as "ran" or "better," but lemmatization maps "ran" to "run" and "better" to "good."

Stemming is faster and simpler; lemmatization is more accurate but slower and needs the part of speech.

**2. When removing stop words helps, and when it hurts.**

**Useful:** stop words such as "the," "is" and "in" appear in almost every text and carry little meaning on their own. Removing them shrinks the vocabulary and the input size, reduces noise, and lets models focus on content words. This helps in tasks such as topic classification, keyword extraction, search indexing and spam filtering.

**Harmful:** when small words carry the meaning. In sentiment analysis, removing "not" turns "not good" into "good," which flips the sentiment. In machine translation, question answering and text generation, words such as "to," "the" and "is" are needed for correct grammar and meaning. Modern Transformer models such as BERT usually keep all words, because they learn from full context.

## Q2: Named Entity Recognition with spaCy

**File:** `q2_named_entity_recognition.py`

The program loads spaCy's English model `en_core_web_sm`, runs it on the given sentence, and prints each entity's text, label and start and end character positions. It also checks that `sentence[start:end]` returns exactly the entity text.

**Output:**

```
Entity text           Label      Start   End
---------------------------------------------
LeBron James          PERSON         1    13
NBA                   ORG           21    24
the United States     GPE           35    52
Laker                 GPE           56    61
MVP                   ORG           84    87
2020                  DATE          91    95

Total entities found: 6
```

Label meanings (from `spacy.explain`): PERSON = people; ORG = companies, agencies, institutions; GPE = countries, cities, states; DATE = dates or periods. The program also prints these meanings next to each entity (see `q2_output.txt`).

The start position of "LeBron James" is 1, not 0, because the given sentence begins with a space. The end position is exclusive, so characters 1–12 are the entity. The small spaCy model makes a few mistakes on this informal sentence: it labels "Laker" as a place (GPE) instead of a team (ORG), labels "MVP" as an organization, and does not detect the second "NBA." A larger model such as `en_core_web_trf`, or a correctly written sentence ("the Lakers"), would give more accurate labels.

### Short Answer Questions

**1. How NER differs from POS tagging.**

**POS tagging** gives every single word its grammatical role, such as noun, verb, adjective or preposition (for example, "won" → VERB, "player" → NOUN). **NER** finds only the words or phrases that name real-world things and classifies them into categories such as PERSON, ORGANIZATION, LOCATION or DATE (for example, "LeBron James" → PERSON, "2020" → DATE). In other words, POS tagging is about grammar and labels every token, while NER is about meaning, labels only named entities, and an entity can span several words.

**2. Two real-world applications of NER.**

- **Financial news and trading:** NER automatically pulls company names, people, amounts and dates from news articles and reports, so analysts and trading systems can track events about a specific company (for example, a CEO change at Apple) as soon as they are published.
- **Search engines and question answering:** search engines identify the people, places and organizations in a query and in web pages, so a search for "Lakers coach 2020" returns pages about that team and shows knowledge panels. Virtual assistants use NER the same way to find the city in "What's the weather in Chicago?"

Other uses include extracting patient names, drugs and diseases from medical records, and routing customer support emails.

## Q3: Scaled Dot-Product Attention

**File:** `q3_scaled_dot_product_attention.py`

The function `scaled_dot_product_attention(Q, K, V)` computes Q·Kᵀ, divides the scores by √d (d = 4, so √d = 2), applies a row-wise softmax, and multiplies the weights by V.

**Output:**

```
Q K^T (raw scores):
 [[2 0]
 [0 2]]

Scaled scores (divided by sqrt(d) = sqrt(4) = 2):
 [[1. 0.]
 [0. 1.]]

1. Attention weights (after softmax):
 [[0.7311 0.2689]
 [0.2689 0.7311]]
   Row sums: [1. 1.]

2. Final output (weights x V):
 [[2.0758 3.0758 4.0758 5.0758]
 [3.9242 4.9242 5.9242 6.9242]]
```

Each query matches its own key best (score 2 vs. 0), so it puts 73% of its attention on its own value row and 27% on the other. The first output row is therefore 0.7311·[1, 2, 3, 4] + 0.2689·[5, 6, 7, 8] = [2.0758, 3.0758, 4.0758, 5.0758].

### Short Answer Questions

**1. Why the attention score is divided by √d.**

The dot product of two d-dimensional vectors adds up d products, so as d grows, the scores grow in size (their variance grows roughly in proportion to d). Very large scores push softmax into saturation: one weight becomes almost 1 and the others almost 0. In that region the gradients become extremely small, and training slows down or stalls. Dividing by √d brings the scores back to a variance of about 1, whatever the dimension, so softmax stays in a range where it produces smooth weights and useful gradients.

**2. How self-attention helps the model understand relationships between words.**

Self-attention compares every word with every other word in the sentence (the query of one word against the keys of all words) and uses the softmax weights to decide how much each word should contribute to the new representation of the current word. Each word's output is therefore a mix of information from the words most relevant to it, wherever they are in the sentence. This lets the model resolve pronouns ("it" → "animal"), choose the right meaning of ambiguous words ("bank" near "river"), and connect subjects with their verbs, even when the words are far apart. With multiple heads, the model learns several of these relationships at the same time.
