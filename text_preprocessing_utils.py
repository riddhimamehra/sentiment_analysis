from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z0-9'/\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
