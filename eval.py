import json
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load data
with open('data.json', encoding="utf8") as file:
    data = json.load(file)

# Preprocess data
lemmatizer = WordNetLemmatizer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize words
        tokens = word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent['tag'])
    
    labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(docs_y)

# One-hot encode labels
onehot_encoder = OneHotEncoder()
encoded_labels = encoded_labels.reshape(len(encoded_labels), 1)
onehot_labels = onehot_encoder.fit_transform(encoded_labels)

# Get unique words and lemmatize them
words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
words = sorted(list(set(words)))

# Load the trained model
model = load_model('model.h5')

# Create word bag for each pattern
X = []
for doc in docs_x:
    bag = [1 if word in doc else 0 for word in words]
    X.append(bag)

X = np.array(X)
Y = np.array(onehot_labels)

# Make predictions
probabilities = model.predict(X)
predictions = np.argmax(probabilities, axis=1)

# Decode one-hot encoded labels
decoded_predictions = label_encoder.inverse_transform(predictions)

# Evaluate the model
accuracy = accuracy_score(docs_y, decoded_predictions)
precision = precision_score(docs_y, decoded_predictions, average='weighted')
f1 = f1_score(docs_y, decoded_predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)