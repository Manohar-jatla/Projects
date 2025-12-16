import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models and tokenizer
encoder_model = tf.keras.models.load_model("models/encoder_model.keras")
decoder_model = tf.keras.models.load_model("models/decoder_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Constants
MAX_LEN = 40
LATENT_DIM = 512
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence with just the <sos> token
    target_seq = np.zeros((1, MAX_LEN))
    target_seq[0, 0] = tokenizer.word_index[SOS_TOKEN]

    decoded_sentence = ""
    for i in range(1, MAX_LEN):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample the next word
        sampled_token_index = np.argmax(output_tokens[0, i - 1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "")

        if sampled_word == EOS_TOKEN or sampled_word == "":
            break

        decoded_sentence += " " + sampled_word

        # Update the target sequence
        target_seq[0, i] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip()

def preprocess_input(question):
    seq = tokenizer.texts_to_sequences([question.lower()])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return seq

if __name__ == "__main__":
    print("💬 Ask a question about TMDB dataset (type 'exit' to quit)")
    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit']:
            break
        input_seq = preprocess_input(question)
        answer = decode_sequence(input_seq)
        print("Bot:", answer)
