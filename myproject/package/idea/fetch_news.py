import requests
import random
from package.idea.config import NEWS_API_KEY

def fetch_sustainability_news():
    themes = ['sustainability', 'environment', 'climate change', 'green energy', 'renewable energy', 'water conservation', 'sustainable development', 'eco-friendly', 'carbon footprint', 'clean energy', 'zero waste', 'global warming', 'climate action']
    
    theme = random.choice(themes)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': theme, 
        'domains': 'timesofindia.indiatimes.com,indianexpress.com,ndtv.com', 
        'apiKey': NEWS_API_KEY, 
        'pageSize': 15
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [(article['title'], article['description']) for article in articles]
    else:
        print("Failed to fetch news. Please try again.")
        return []