import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from googletrans import Translator
import pandas as pd
from etnltk import Amharic

translator = Translator()


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def clean_text(text):
    if pd.isnull(text):
        return text
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'^\w\s', '', text)
    return text


def clean_amharic_text(text):
    return Amharic(text)


def remove_english_words(text):
    if pd.isnull(text):
        return text
    cleaned = re.sub(r'\b[a-zA-Z]+\b', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def remove_amharic_words(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    # Regex pattern to match Amharic characters
    amharic_pattern = re.compile(
        r'\b[\u1200-\u137F\u1308-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+\b')
    cleaned_text = amharic_pattern.sub('', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def extract_sender_and_msg(text):
    if pd.isnull(text):
        return pd.Series([None, None])
    match = re.match(r'^(\w\s\.\-@]+):\s*(.+)', text)
    if match:
        return pd.Series([match.group(1).strip(), match.group(2).strip()])
    else:
        return pd.Series([None, text.strip()])


def clean_message(msg):
    if pd.isnull(msg):
        return msg
    cleaned_msg = re.sub(r'http\S+', '', msg)  # Remove URLs
    cleaned_msg = re.sub(r'@\w+', '', cleaned_msg)  # Remove mentions
    cleaned_msg = re.sub(r'#\w+', '', cleaned_msg)  # Remove hashtags
    # Remove punctions except Amharic
    cleaned_msg = re.sub(r'[^\w\s:።መለያየትም፣፤፦]', '', cleaned_msg)
    # Remove extra spaces
    cleaned_msg = re.sub(r'\s+', ' ', cleaned_msg).strip()
    cleaned_msg = remove_english_words(cleaned_msg)
    return cleaned_msg


def translate_to_amharic(text):
    try:
        translation = translator.translate(text, src='auto', dest='am')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text
