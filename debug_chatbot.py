import json
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and test
with open('chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load intents
with open('intents.json', 'r') as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Test messages
test_messages = [
    "Hello",
    "What's the weather in Tokyo?",
    "seattle weather",
    "Hi there"
]

print("Testing intent classification:")
print("-" * 50)

for message in test_messages:
    processed = preprocess_text(message)
    prediction = model.predict([processed])[0]
    probabilities = model.predict_proba([processed])[0]
    confidence = probabilities.max()
    
    print(f"Original: '{message}'")
    print(f"Processed: '{processed}'")
    print(f"Predicted intent: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    print(f"All probabilities: {dict(zip(model.classes_, probabilities))}")
    print("-" * 30)
