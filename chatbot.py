### Model Preprocessing

import os
import yaml
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.utils import pad_sequences
import pickle

DATA_PATH = "data"

questions = []
answers = []

# Parse YAML files
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".yml"):
        with open(os.path.join(DATA_PATH, filename), 'r', encoding='utf-8') as file:
            docs = yaml.safe_load(file)
            for dialog in docs['conversations']:
                if len(dialog) >= 2:
                    q = dialog[0]
                    a = " ".join(dialog[1:])  # merge multiple answers
                    questions.append(q)
                    answers.append("<START> " + a + " <END>")

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

VOCAB_SIZE = len(tokenizer.word_index) + 1

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Sequence padding
max_len = 20  # you can adjust based on data analysis
encoder_input_data = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=max_len, padding='post')
decoder_input_data = pad_sequences(tokenizer.texts_to_sequences(answers), maxlen=max_len, padding='post')

# Remove <START> from decoder_output_data
decoder_output_data = []
for ans in answers:
    ans_split = ans.split()
    ans_split = ans_split[1:]  # remove <START>
    decoder_output_data.append(" ".join(ans_split))
decoder_output_data = pad_sequences(tokenizer.texts_to_sequences(decoder_output_data), maxlen=max_len, padding='post')

# Convert to categorical
from tensorflow.keras.utils import to_categorical
decoder_output_data = to_categorical(decoder_output_data, num_classes=VOCAB_SIZE)

print("Data preprocessing complete.")


### Building the model

from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense

EMBEDDING_DIM = 100
LSTM_UNITS = 256

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(LSTM_UNITS, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

### Model Training

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    batch_size=64,
    epochs=150,
    validation_split=0.2
)

model.save("chatbot_model.keras")
