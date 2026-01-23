# 🧠 Byte Pair Encoding (BPE) Tokenizer — From Scratch

This repository contains a **from-scratch implementation of a Byte Pair Encoding (BPE) tokenizer**, built step by step to demonstrate how modern NLP tokenizers (like those used in GPT-style models) work internally.

The project walks through:

* What tokenization is
* Why naive word tokenization fails
* How BPE solves vocabulary explosion
* How to train a BPE tokenizer
* How to tokenize unseen text using learned merge rules

---

## 📌 What is Tokenization?

Tokenization is the process of breaking raw text into smaller units called **tokens**, which are the basic inputs to NLP models.

A naive approach splits text into words, but this leads to:

* Very large vocabularies
* Poor handling of rare or unseen words
* Increased memory and compute cost

---

## 🔹 Why Byte Pair Encoding (BPE)?

**Byte Pair Encoding (BPE)** is a **subword tokenization algorithm** that:

* Starts from characters
* Iteratively merges the most frequent adjacent symbol pairs
* Builds a compact and flexible vocabulary
* Handles unseen words by decomposing them into known subwords

This makes BPE:

* Memory efficient
* Robust to rare words
* Suitable for large language models (e.g., GPT family)

---

## 🧪 Training Data

Two types of corpora are demonstrated:

1. **Toy corpus**
   A short excerpt from *Alice’s Adventures in Wonderland* for conceptual clarity.

2. **Real-world corpus**
   Wikipedia text retrieved dynamically using `langchain-community`’s `WikipediaRetriever`.

---

## ⚙️ Project Pipeline

### 1. Text Extraction

```python
ExtractContentFromWikipedia(query="Nigerian History")
```

* Retrieves Wikipedia pages
* Concatenates content
* Converts text to lowercase

---

### 2. Initial Tokenization

```python
r"\w+|[^\s\w]+"
```

* Captures words and punctuation separately
* Preserves punctuation as tokens

---

### 3. Word Frequency Counting

```python
Counter(init_word_list)
```

* Counts how often each token appears
* Frequencies drive merge decisions

---

### 4. Initial Corpus Representation

Each word is represented as:

```text
['c', 'h', 'a', 'r', 's', '</end>']
```

The `</end>` token:

* Prevents merges across word boundaries
* Allows learning suffixes like `ing</end>`, `s</end>`

---

### 5. Vocabulary Initialization

The initial vocabulary contains:

* All unique characters
* The end-of-word token

---

### 6. BPE Training Loop

For each merge iteration:

1. Count all adjacent symbol pairs
2. Select the most frequent pair
3. Merge it into a new subword token
4. Update:

   * Corpus representation
   * Vocabulary
   * Learned merge rules

```python
best_pair = max(pair_count, key=pair_count.get)
new_token = "".join(best_pair)
```

---

### 7. Learned Outputs

After training:

* `final_unique_token`: learned vocabulary
* `final_learned_merges`: ordered merge rules
* `final_wrd_split`: final corpus representation

---

## 🧾 Tokenizing Unseen Text

The tokenizer can encode **previously unseen words** using learned subwords.

Example:

```python
test_sentence = "lower newest wider"
```

Output:

```text
[['low', 'er'], ['ne', 'west'], ['wi', 'der']]
```

This demonstrates:

* Subword reuse
* Generalization beyond training data

---

## 📦 Installation

```bash
pip install -qU langchain-community wikipedia
```

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies
3. Run the notebook or Python script
4. Train the tokenizer:

```python
final_unique_token, final_learned_merges, final_wrd_split = TrainTokenizer()
```

5. Encode new text:

```python
encode_sentence("your text here", final_learned_merges)
```

---

## 🧠 Key Takeaways

* BPE balances vocabulary size and expressiveness
* Subword tokenization enables robust handling of unseen words
* This tokenizer mirrors the **core logic used in GPT-style models**
* Tokenization is a critical foundation for building LLMs

---

## 🚀 What’s Next?

The next step after tokenization is **sequence modeling**.

Following GPT-style architectures, the natural progression is:

* Implementing **Transformer blocks**
* Understanding **self-attention**
* Building a **text generation LLM** based on “Attention Is All You Need”

---

## 📜 License

This project is for **educational and research purposes**.
