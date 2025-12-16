import json
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Settings
MAX_LEN = 40
MAX_VOCAB_SIZE = 15000
DATA_PATH = "data/qna_pairs.json"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_prepare_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    questions = []
    answers = []

    for q, a in pairs:
        q = clean_text(q)
        a = clean_text(a)
        # Add special tokens for decoder
        answers.append(f"<sos> {a} <eos>")
        questions.append(q)

    print(f"Total cleaned pairs: {len(questions)}")

    return questions, answers

def tokenize_and_pad(questions, answers):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<unk>", filters='')
    tokenizer.fit_on_texts(questions + answers)

    # Convert to sequences
    q_seq = tokenizer.texts_to_sequences(questions)
    a_seq = tokenizer.texts_to_sequences(answers)

    # Pad sequences
    q_pad = pad_sequences(q_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    a_pad = pad_sequences(a_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    word_index = tokenizer.word_index
    index_word = {i: w for w, i in word_index.items()}

    print(f"Vocabulary size: {len(word_index)}")

    return q_pad, a_pad, tokenizer, word_index, index_word

def save_preprocessed_data(q_pad, a_pad, tokenizer, word_index, index_word):
    os.makedirs("data", exist_ok=True)
    np.save("data/encoder_input.npy", q_pad)
    np.save("data/decoder_input.npy", a_pad[:, :-1])  # decoder input (excluding <eos>)
    np.save("data/decoder_target.npy", a_pad[:, 1:])  # decoder target (excluding <sos>)

    with open("data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("data/word_index.pkl", "wb") as f:
        pickle.dump(word_index, f)
    with open("data/index_word.pkl", "wb") as f:
        pickle.dump(index_word, f)

if __name__ == "__main__":
    questions, answers = load_and_prepare_data()
    q_pad, a_pad, tokenizer, word_index, index_word = tokenize_and_pad(questions, answers)

    # ✅ Check that <sos> is in the tokenizer
    print("<sos> in tokenizer.word_index?", "<sos>" in tokenizer.word_index)
    print("Index of <sos>:", tokenizer.word_index.get("<sos>"))

    save_preprocessed_data(q_pad, a_pad, tokenizer, word_index, index_word)
