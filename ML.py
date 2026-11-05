import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# Load the text dataset
with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()  # Convert to lowercase for uniformity

# Tokenize the text (assign a unique integer ID to every unique word)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Total number of unique words in your vocabulary (+1 accounts for padding index)
total_words = len(tokenizer.word_index) + 1
print(f"Total Unique Words: {total_words}")
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        # Create n-gram sequences: e.g., "the cat sat" -> ["the", "the cat"]
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Find the maximum sequence length for padding later
max_sequence_len = max([len(seq) for seq in input_sequences])

# Pad all sequences so they are the same length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split into features (X) and label (y)
# X = all words except the last one, y = the last word (target)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode the target word output across the total vocabulary size
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
model = Sequential([
    # Embedding layer transforms word indices into dense vectors of fixed size (e.g., 100)
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    
    # LSTM layer to capture context over sequential inputs
    LSTM(150, return_sequences=False),
    Dropout(0.2), # Prevents overfitting
    
    # Output layer with Softmax activation to output probabilities for every word in vocabulary
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=50, verbose=1)
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    for _ in range(next_words):
        # Tokenize the seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad it to match training shape
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict the probability distribution of the next word
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Map the predicted index back to the actual word string
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
                
        # Append the predicted word to the seed text
        seed_text += " " + output_word
        
    return seed_text

# Test your next-word generator!
print(generate_text("the detective", next_words=5, model=model, tokenizer=tokenizer, max_sequence_len=max_sequence_len))