"""HTML parsing utilities"""
import requests
from bs4 import BeautifulSoup
import re

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def parse_url(url: str) -> dict:
    """Fetch and parse a URL"""
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()
        
        # Get main content
        main_content = soup.find('article') or soup.find('main') or soup
        paragraphs = main_content.find_all('p')
        body_text = ' '.join([p.get_text().strip() for p in paragraphs])
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        
        return {
            'title': title,
            'body_text': body_text,
            'url': url
        }
    
    except requests.RequestException as e:
        return {'error': f"Failed to fetch URL: {str(e)}"}
    except Exception as e:
        return {'error': f"Parsing error: {str(e)}"}