import re
from typing import Dict, List
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from googletrans import Translator
import pandas as pd
from etnltk import Amharic
from datetime import datetime

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


def clean_amharic_text(text: str) -> str:
    """Clean and normalize Amharic text"""
    if not text or pd.isna(text):
        return ""

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove URLs
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)

    # Normalize Amharic punctuation
    text = text.replace('፡', ':')
    text = text.replace('።', '.')
    text = text.replace('፣', ',')
    text = text.replace('፤', ';')
    text = text.replace('፥', ':')
    text = text.replace('፦', ':')

    # Remove emoji patterns (basic)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    return text.strip()


def tokenize_amharic(text: str) -> List[str]:
    """Tokenize Amharic text"""
    amharic_range = r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]'
    if not text:
        return []

    # Split by whitespace and punctuation while preserving Amharic characters
    tokens = re.findall(r'\S+', text)

    # Further split tokens that contain mixed scripts
    processed_tokens = []
    for token in tokens:
        # If token contains both Amharic and non-Amharic, try to split
        if re.search(amharic_range, token) and re.search(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\s]', token):
            # Split mixed tokens
            subtokens = re.findall(
                r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]+|[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\s]+', token)
            processed_tokens.extend([t for t in subtokens if t.strip()])
        else:
            if token.strip():
                processed_tokens.append(token)

    return processed_tokens


def is_amharic_dominant(text: str) -> bool:
    """Check if text is predominantly Amharic"""
    amharic_range = r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]'
    if not text:
        return False

    amharic_chars = len(re.findall(amharic_range, text))
    total_chars = len(re.sub(r'\s+', '', text))

    if total_chars == 0:
        return False

    return (amharic_chars / total_chars) > 0.5


def normalize_numbers(text: str) -> str:
    """Normalize number representations"""
    # Convert Amharic numbers to Arabic numerals if needed
    amharic_to_arabic = {
        '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
        '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10'
    }

    for amharic, arabic in amharic_to_arabic.items():
        text = text.replace(amharic, arabic)

    return text


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from amharic text"""
    price_patterns = [
        r'ዋጋ\s*[:፦]?\s*\d+',
        r'\d+\s*ብር',
        r'በ\s*\d+\s*ብር',
        r'ብር\s*\d+',
        r'\d+\s*birr',
        r'price\s*[:፦]?\s*\d+'
    ]
    location_patterns = [
        'አዲስ አበባ', 'አዲስ', 'አበባ', 'መገናኛ', 'ቦሌ', 'ጣና', 'ወሎ ሰፈር', 'ቢሾፍቱ', 'ሀዋሳ', 'ባህር ዳር', 'መቀሌ', 'አርባ ምንጭ', 'ጎንደር', 'ደሴ'
    ]
    product_indicators = ['ዕቃ', 'ምርት', 'አይነት', 'ሸቀጣሸቀጥ']
    entities = {
        'price_hints': [],
        'location_hints': [],
        'product_hints': []
    }
    # extract price hints
    for pattern in price_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities['price_hints'].append(match.group())
    # extract location hints
    for location in location_patterns:
        if location.lower() in text.lower():
            entities['location_hints'].append(location)

    # extract product hints
    for product in product_indicators:
        pattern = rf'{product}\s+([^\s]+(?:\s+[^\s]+)?)'
        matches = re.finditer(pattern, text)
        for match in matches:
            entities['product_hints'].append(match.group(1))

    return entities


def prepare_dataset(df):
    df_processed = df.copy()
    df_processed['clean_message'] = df_processed['Message'].apply(
        clean_amharic_text)
    df_processed['is_amharic'] = df_processed['clean_message'].apply(
        is_amharic_dominant)
    df_processed['entity_hints'] = df_processed['clean_message'].apply(
        extract_entities)
    df_processed['tokens'] = df_processed['clean_message'].apply(
        tokenize_amharic)
    df_processed['tokens_count'] = df_processed['tokens'].apply(len)
    df_processed['process_time'] = datetime.now().isoformat()
    df_processed['has_price_hints'] = df_processed['entity_hints'].apply(
        lambda x: len(x['price_hints']) > 0)
    df_processed['has_location_hints'] = df_processed['entity_hints'].apply(
        lambda x: len(x['location_hints']) > 0)
    df_processed['has_product_hints'] = df_processed['entity_hints'].apply(
        lambda x: len(x['product_hints']) > 0)

    df_processed = df_processed[df_processed['tokens_count'] >= 3]

    return df_processed
