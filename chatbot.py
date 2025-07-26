import json
import re
import random
import pickle
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd

# Load environment variables
load_dotenv()

class WeatherChatbot:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.intents = {}
        self.city_db = {}
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        
        if not self.api_key:
            print("⚠️ Warning: OpenWeatherMap API key not found in environment variables!")
            print("Please ensure your .env file contains: OPENWEATHER_API_KEY=your_key_here")
        
        self.load_intents()
        self.load_city_database()
        self.download_nltk_data()
        self.load_or_train_model()
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def load_intents(self):
        """Load intents from JSON file"""
        try:
            with open('intents.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.intents = {intent['tag']: intent for intent in data['intents']}
            print(f"✅ Loaded {len(self.intents)} intents")
        except FileNotFoundError:
            print("❌ Error: intents.json file not found!")
            print("Please create intents.json with your training data.")
            exit(1)
    
    def load_city_database(self):
        """Load city database for improved location matching"""
        try:
            with open('city_list.json', 'r', encoding='utf-8') as f:
                cities = json.load(f)
            
            self.city_db = {}
            for city in cities:
                name = city.get('name', '').strip()
                if name and len(name) >= 2:
                    # Create multiple lookup keys for better matching
                    clean_name = self.clean_city_name(name)
                    
                    city_info = {
                        'id': city.get('id'),
                        'name': name,
                        'country': city.get('country', ''),
                        'coord': city.get('coord', {}),
                        'state': city.get('state', '')
                    }
                    
                    # Store under multiple keys for flexible lookup
                    self.city_db[clean_name.lower()] = city_info
                    self.city_db[name.lower()] = city_info
                    
            print(f"✅ Loaded {len(self.city_db)} cities into database")
        except FileNotFoundError:
            print("⚠️ City database (city_list.json) not found. Using text-based location extraction only.")
            self.city_db = {}
    
    def clean_city_name(self, name):
        """Clean city name for better matching"""
        # Remove special characters but keep spaces and hyphens
        cleaned = re.sub(r'[^\w\s-]', '', name)
        return cleaned.strip()
    
    def preprocess_text(self, text):
        """Enhanced preprocessing that retains more context for better intent recognition"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        
        # Use minimal stop words to retain more context
        minimal_stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in minimal_stop_words and len(token) > 1]
        return ' '.join(tokens)
    
    def prepare_training_data(self):
        """Prepare training data from intents"""
        X = []
        y = []
        
        for tag, intent in self.intents.items():
            for pattern in intent['patterns']:
                processed_pattern = self.preprocess_text(pattern)
                X.append(processed_pattern)
                y.append(tag)
        
        return X, y
    
    def train_model(self):
        """Train the intent classification model with optimized parameters"""
        print("Training model with global city data...")
        X, y = self.prepare_training_data()
        
        print(f"Training with {len(X)} patterns across {len(set(y))} intents")
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=1,
                analyzer='word'
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=3000,
                C=4.0,
                class_weight='balanced'
            ))
        ])
        
        self.model.fit(X, y)
        
        # Cross-validation
        scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Model accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        # Save model
        with open('chatbot_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
        print("Model trained and saved successfully!")
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            with open('chatbot_model.pkl', 'rb') as file:
                self.model = pickle.load(file)
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("No existing model found. Training new model...")
            self.train_model()
    
    def predict_intent(self, message):
        """Predict intent from user message"""
        processed_message = self.preprocess_text(message)
        prediction = self.model.predict([processed_message])[0]
        confidence = self.model.predict_proba([processed_message]).max()
        
        return prediction, confidence
    
    def extract_location(self, message):
        """Enhanced location extraction with city database support"""
        message = message.strip().lower()
        
        # First, try to find city in database
        city_match = self.find_city_in_database(message)
        if city_match:
            return city_match
        
        # Fallback to pattern-based extraction
        location_patterns = [
            r'weather in ([a-z\s\-\']+)',                   # "weather in new york"
            r'weather for ([a-z\s\-\']+)',                  # "weather for paris" 
            r'temperature in ([a-z\s\-\']+)',               # "temperature in london"
            r'temp in ([a-z\s\-\']+)',                      # "temp in delhi"
            r'climate in ([a-z\s\-\']+)',                   # "climate in sydney"
            r'how.*weather.*in ([a-z\s\-\']+)',             # "how's weather in miami"
            r'what.*weather.*in ([a-z\s\-\']+)',            # "what's weather in dubai"
            r'how.*temperature.*in ([a-z\s\-\']+)',         # "how's temperature in cairo"
            r'^([a-z\s\-\']+)\s+weather$',                  # "london weather"
            r'^([a-z\s\-\']+)\s+city\s+weather$',          # "new york city weather"
            r'^([a-z\s\-\']+)\s+temp(?:erature)?$',        # "paris temp"
            r'^([a-z\s\-\']+)\s+climate$',                  # "tokyo climate"
            r'weather\s+([a-z\s\-\']+)$',                   # "weather london"
            r'temp(?:erature)?\s+([a-z\s\-\']+)$',         # "temperature paris"
            r'climate\s+([a-z\s\-\']+)$',                   # "climate sydney"
            r'^([a-z\-\']{3,}(?:\s+[a-z\-\']+){0,3})$',    # standalone city names
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message)
            if match:
                location = match.group(1).strip()
                location = re.sub(r'\b(city|weather|temp|temperature|climate|forecast)\b', '', location).strip()
                
                # Filter out common non-location words
                excluded_words = {
                    'the', 'and', 'for', 'you', 'are', 'how', 'what', 'yes', 'now', 'today',
                    'tomorrow', 'here', 'there', 'this', 'that', 'will', 'can', 'could',
                    'would', 'should', 'may', 'might', 'hello', 'hi', 'hey', 'thanks',
                    'thank', 'please', 'good', 'bad', 'nice', 'great', 'ok', 'okay'
                }
                
                if (len(location) >= 3 and 
                    location not in excluded_words and 
                    not location.isdigit() and
                    len(location.replace(' ', '')) >= 3):
                    
                    # Try to find in database
                    db_match = self.find_city_in_database(location)
                    return db_match if db_match else location.title()
        
        return None
    
    def find_city_in_database(self, query):
        """Find city in the loaded database"""
        if not self.city_db:
            return None
        
        query = query.lower().strip()
        
        # Exact match
        if query in self.city_db:
            city_info = self.city_db[query]
            return city_info['name']
        
        # Fuzzy matching for close matches
        for city_key, city_info in self.city_db.items():
            if query in city_key or city_key in query:
                if abs(len(query) - len(city_key)) <= 2:  # Allow small length differences
                    return city_info['name']
        
        return None
    
    def get_weather_data(self, city):
        """Fetch weather data with enhanced error handling and city ID support"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        # Try to get city ID for more accurate API call
        city_id = self.get_city_id(city)
        
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        if city_id:
            params = {
                "id": city_id,
                "appid": self.api_key,
                "units": "metric"
            }
        else:
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric"
            }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": f"City '{city}' not found. Please check the spelling or try a different format."}
            elif response.status_code == 401:
                return {"error": "Invalid API key. Please check your OpenWeatherMap API key."}
            elif response.status_code == 429:
                return {"error": "API rate limit exceeded. Please try again later."}
            else:
                return {"error": f"Weather service error (Code: {response.status_code})"}
                
        except requests.exceptions.Timeout:
            return {"error": "Weather service is taking too long to respond. Please try again."}
        except requests.exceptions.ConnectionError:
            return {"error": "Unable to connect to weather service. Please check your internet connection."}
        except requests.exceptions.RequestException as e:
            return {"error": f"Weather service error: {str(e)}"}
    
    def get_city_id(self, city_name):
        """Get city ID from database for more accurate API calls"""
        if not self.city_db:
            return None
        
        city_key = city_name.lower()
        if city_key in self.city_db:
            return self.city_db[city_key]['id']
        
        return None
    
    def format_weather_response(self, weather_data):
        """Format weather data into readable response with rich formatting"""
        if "error" in weather_data:
            return f"❌ {weather_data['error']}"
        
        try:
            city = weather_data['name']
            country = weather_data['sys']['country']
            temp = weather_data['main']['temp']
            feels_like = weather_data['main']['feels_like']
            humidity = weather_data['main']['humidity']
            pressure = weather_data['main']['pressure']
            description = weather_data['weather'][0]['description'].title()
            wind_speed = weather_data.get('wind', {}).get('speed', 'N/A')
            visibility = weather_data.get('visibility', 'N/A')
            
            # Enhanced weather emojis
            weather_emojis = {
                'clear': '☀️',
                'clouds': '☁️',
                'rain': '🌧️',
                'drizzle': '🌦️',
                'thunderstorm': '⛈️',
                'snow': '❄️',
                'mist': '🌫️',
                'fog': '🌫️',
                'haze': '🌫️',
                'dust': '💨',
                'sand': '💨',
                'ash': '🌋',
                'squall': '💨',
                'tornado': '🌪️'
            }
            
            weather_main = weather_data['weather'][0]['main'].lower()
            emoji = weather_emojis.get(weather_main, '🌤️')
            
            # Format visibility
            visibility_str = f"{visibility/1000:.1f} km" if visibility != 'N/A' else 'N/A'
            
            # Add local time if available
            timezone_offset = weather_data.get('timezone', 0)
            local_time = datetime.utcnow().timestamp() + timezone_offset
            local_time_str = datetime.fromtimestamp(local_time).strftime('%H:%M')
            
            response = f"""{emoji} Current weather in {city}, {country}:
🌡️ Temperature: {temp}°C (feels like {feels_like}°C)
☁️ Condition: {description}
💧 Humidity: {humidity}%
🌬️ Wind Speed: {wind_speed} m/s
📊 Pressure: {pressure} hPa
👁️ Visibility: {visibility_str}
🕐 Local Time: {local_time_str}"""
            
            return response
            
        except KeyError as e:
            return f"❌ Error parsing weather data: Missing field {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error formatting weather data: {str(e)}"
    
    def get_response(self, message):
        """Generate response based on user message with improved logic"""
        intent, confidence = self.predict_intent(message)
        
        # Dynamic confidence threshold based on message complexity
        threshold = 0.45 if len(message.split()) <= 2 else 0.55
        
        if confidence < threshold:
            return "I'm not sure I understand. I can help you with weather information for any city worldwide! Try asking about the weather in a specific city."
        
        # Handle all weather-related intents
        if intent in ["weather_location", "weather_current"]:
            location = self.extract_location(message)
            if location:
                weather_data = self.get_weather_data(location)
                return self.format_weather_response(weather_data)
            else:
                return "I'd love to help with weather information! Which city would you like to know about?"
        
        # Handle other intents
        elif intent in self.intents:
            return random.choice(self.intents[intent]['responses'])
        
        return "I'm here to help with weather information for cities worldwide! Ask me about any city's weather."
    
    def chat(self):
        """Main chat loop with enhanced user experience"""
        print("=" * 70)
        print("🌍 Global Weather Chatbot with Comprehensive City Database!")
        print("I can get weather information for any city in the world!")
        print("=" * 70)
        print("Examples:")
        print("• Just city name: 'London', 'Mumbai', 'Begusarai'")
        print("• With weather: 'Tokyo weather', 'weather in Paris'")
        print("• Questions: 'What's the weather in New York?'")
        print("• Temperature: 'Temperature in Delhi'")
        print("• Type 'quit' to exit")
        print("=" * 70)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("Bot: Goodbye! Stay safe and have wonderful weather! 👋🌤️")
                break
            
            if not user_input:
                continue
                
            response = self.get_response(user_input)
            print(f"Bot: {response}")

def main():
    print("=== Global Weather Chatbot with City Database ===")
    print("Initializing with comprehensive city data...")
    
    try:
        chatbot = WeatherChatbot()
        chatbot.chat()
    except KeyboardInterrupt:
        print("\n\nBot: Goodbye! 👋")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure you have:")
        print("1. intents.json with training data")
        print("2. city_list.json with OpenWeatherMap city database")
        print("3. .env file with OPENWEATHER_API_KEY")

if __name__ == "__main__":
    main()
