# train.py

import numpy as np
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from model import create_seq2seq_model, create_inference_models

# Load data
encoder_input = np.load("data/encoder_input.npy")
decoder_input = np.load("data/decoder_input.npy")
decoder_input = np.pad(decoder_input, ((0, 0), (0, 1)), mode='constant', constant_values=0)
decoder_target = np.load("data/decoder_target.npy")
decoder_target = np.pad(decoder_target, ((0, 0), (0, 1)), mode='constant', constant_values=0)


with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Hyperparameters
VOCAB_SIZE = 15000
EMBED_DIM = 512
LATENT_DIM = 512
MAX_LEN = 40
BATCH_SIZE = 64
EPOCHS = 50


# Create model
model = create_seq2seq_model(VOCAB_SIZE, EMBED_DIM, LATENT_DIM, MAX_LEN)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Save best weights
checkpoint = ModelCheckpoint("models/seq2seq_weights.h5", save_best_only=True, monitor="val_loss", verbose=1)

# Train the model
history = model.fit(
    [encoder_input, decoder_input],
    decoder_target,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=[checkpoint]
)

# Save encoder and decoder inference models
encoder_model, decoder_model = create_inference_models(model)
encoder_model.save("models/encoder_model.keras")
decoder_model.save("models/decoder_model.keras")



# Save tokenizer again just in case
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Training complete. Models saved in /models.")
