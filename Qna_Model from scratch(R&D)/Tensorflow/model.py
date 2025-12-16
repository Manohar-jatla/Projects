# model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Hyperparameters
VOCAB_SIZE = 15000
EMBED_DIM = 512
LATENT_DIM = 512
MAX_LEN = 40

def create_seq2seq_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM, max_len=MAX_LEN):
    # Encoder
    encoder_inputs = Input(shape=(max_len,), name="encoder_inputs")
    enc_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name="encoder_embedding")(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_len,), name="decoder_inputs")
    dec_emb_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Full training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# For inference model
def create_inference_models(model):
    # Encoder inference model
    encoder_inputs = model.input[0]  # encoder_inputs
    encoder_embedding = model.get_layer("encoder_embedding")(encoder_inputs)
    _, state_h_enc, state_c_enc = model.get_layer("encoder_lstm")(encoder_embedding)
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    # Decoder inference model
    decoder_inputs = model.input[1]  # decoder_inputs
    decoder_state_input_h = Input(shape=(LATENT_DIM,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(LATENT_DIM,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding = model.get_layer("decoder_embedding")(decoder_inputs)
    decoder_lstm = model.get_layer("decoder_lstm")
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.get_layer("decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

if __name__ == "__main__":
    model = create_seq2seq_model()
    model.summary()
