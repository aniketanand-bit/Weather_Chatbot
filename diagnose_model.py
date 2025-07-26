import json
import re

def process_openweather_cities(input_file, output_file):
    """Convert OpenWeatherMap city list to chatbot training data"""
    
    # Read the city data
    with open(input_file, 'r', encoding='utf-8') as f:
        cities_data = json.load(f)
    
    # Extract city names and clean them
    city_names = []
    country_codes = {}
    
    for city in cities_data:
        name = city.get('name', '').strip()
        country = city.get('country', '').strip()
        
        if name and len(name) >= 3:  # Filter out very short names
            # Clean special characters for training data
            clean_name = re.sub(r'[^\w\s-]', '', name)
            clean_name = clean_name.strip()
            
            if clean_name and clean_name not in city_names:
                city_names.append(clean_name)
                country_codes[clean_name] = country
    
    # Generate training patterns
    patterns = []
    
    # Add standalone city names (first 1000 to avoid overwhelming the model)
    patterns.extend(city_names[:1000])
    
    # Add weather query variations for top 200 cities
    top_cities = city_names[:200]
    for city in top_cities:
        patterns.extend([
            f"{city} weather",
            f"weather in {city}",
            f"temperature in {city}",
            f"weather for {city}",
            f"how's weather in {city}",
            f"what's weather in {city}",
            f"{city} temperature",
            f"{city} forecast"
        ])
    
    # Create comprehensive training data
    training_data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": [
                    "Hi", "Hello", "Hey", "Good morning", "Good afternoon", "Good evening",
                    "Greetings", "What's up", "How are you", "Hi there", "Hello there",
                    "Hey there", "Good day", "Hiya", "Howdy"
                ],
                "responses": [
                    "Hello! I'm here to help you with weather information. How can I assist you today?",
                    "Hi there! I can provide weather updates for any city. What would you like to know?",
                    "Hey! Great to see you. Ask me about the weather anywhere in the world!",
                    "Good day! I'm your weather assistant. Feel free to ask about current conditions or forecasts."
                ]
            },
            {
                "tag": "weather_location",
                "patterns": patterns,
                "responses": [
                    "Let me get the weather information for that location!",
                    "I'll fetch the current weather data for you right away!",
                    "Getting the latest weather information for that city..."
                ]
            },
            {
                "tag": "weather_current",
                "patterns": [
                    "What's the weather like", "Current weather", "How's the weather",
                    "Weather today", "Today's weather", "What's it like outside",
                    "Is it sunny", "Is it raining", "Weather conditions", "Current conditions",
                    "How's the weather today", "What's the temperature", "Is it hot", "Is it cold"
                ],
                "responses": [
                    "I'd be happy to help with weather information! Could you please tell me which city you'd like to know about?",
                    "Sure! I can get current weather conditions for you. Which location are you interested in?",
                    "I can provide current weather data. Please specify the city or location you want to check."
                ]
            },
            {
                "tag": "thanks",
                "patterns": [
                    "Thanks", "Thank you", "Thanks a lot", "Thank you so much",
                    "I appreciate it", "Cheers", "Thanks for your help",
                    "Thank you for the information", "Much appreciated", "Great, thanks"
                ],
                "responses": [
                    "You're welcome! Happy to help with weather info anytime!",
                    "Glad I could help! Feel free to ask about weather conditions anytime.",
                    "You're very welcome! I'm here whenever you need weather updates."
                ]
            },
            {
                "tag": "goodbye",
                "patterns": [
                    "Bye", "Goodbye", "See you later", "See you",
                    "Talk to you later", "Catch you later", "Farewell",
                    "Until next time", "Take care", "Have a good day", "Quit", "Exit"
                ],
                "responses": [
                    "Goodbye! Stay safe and have a great day! 👋",
                    "See you later! Don't forget to check the weather before you go out!",
                    "Take care! I'll be here whenever you need weather updates."
                ]
            }
        ]
    }
    
    # Save the processed training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed {len(city_names)} cities")
    print(f"✅ Generated {len(patterns)} training patterns")
    print(f"✅ Saved to {output_file}")
    
    return len(city_names), len(patterns)

# Usage
if __name__ == "__main__":
    # Assuming your city data is in 'city_list.json'
    process_openweather_cities('city_list.json', 'global_weather_intents.json')
