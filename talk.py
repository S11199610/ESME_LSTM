import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Constants ===
LSTM_UNITS = 256
EMBEDDING_DIM = 100
MAX_LEN = 10 # Can stop chat rambling if it occurs

# === Load tokenizer and trained model ===
model = load_model("chatbot_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

VOCAB_SIZE = len(tokenizer.word_index) + 1

# === Rebuild Encoder Model ===
encoder_inputs = model.input[0]
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
_, state_h_enc, state_c_enc = LSTM(LSTM_UNITS, return_state=True)(encoder_embedding)
encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(encoder_inputs, encoder_states)

# === Rebuild Decoder Model ===
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(decoder_inputs)

decoder_state_input_h = Input(shape=(LSTM_UNITS,))
decoder_state_input_c = Input(shape=(LSTM_UNITS,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)

decoder_dense = Dense(VOCAB_SIZE, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + [state_h, state_c]
)

# === Chat Functions ===
def str_to_tokens(sentence: str):
    tokens = tokenizer.texts_to_sequences([sentence])
    return pad_sequences(tokens, maxlen=MAX_LEN, padding='post')

def decode_sequence(input_text):
    input_seq = str_to_tokens(input_text)
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get('start', 1)

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None

        for word, index in tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word == 'end' or len(decoded_sentence.split()) > MAX_LEN:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# === Start Chatting ===
print("ESME is online! (type 'exit' or 'quit' to end)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Peace out! 👋")
        break
    response = decode_sequence(user_input.lower())
    print("Chatbot:", response)
