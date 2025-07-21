import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('labeling_log.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK tokenizer (run once if needed)
try:
    nltk.download('punkt', quiet=True)
    logger.info("NLTK punkt tokenizer downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK punkt tokenizer: {str(e)}")
    raise

# Load dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset from {file_path}, shape: {df.shape}")
        # Filter out rows where Message is "መልዕክት የለም" or empty
        messages = df['Message'].dropna().tolist()
        messages = [msg for msg in messages if msg != "መልዕክት የለም"]
        logger.info(f"Filtered {len(messages)} valid messages from dataset")
        return messages
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {str(e)}")
        raise

# Helper function to detect entities in Amharic text
def label_entities(message):
    try:
        if not message.strip():  # Check for empty or whitespace-only message
            logger.error(f"Error labeling message: Empty message provided")
            return [], []
        
        tokens = word_tokenize(message)  # Tokenize the message
        labels = ['O'] * len(tokens)  # Initialize all tokens as 'O'
        logger.debug(f"Tokenized message: {tokens[:10]}... (total {len(tokens)} tokens)")

        # Define patterns and keywords based on the dataset
        product_keywords = [
            'መዓዛን', 'የሚሰጥ', 'ማሞቂያ', 'ኮንቴነር', 'ማስጫ', 'ቦርሳ', 'መደገፊያ',
            'ፓይስትራ', 'ማሽን', 'ፍላሽ', 'ላይት', 'ፓወር', 'ባንክ', 'ወተት', 'ልብስ',
            'ማስቀመጫ', 'የሚያሳይ', 'ክሬም', 'የሚስብ', 'የሚያገለግል', 'ጥላ', 'ጁስ', 'መፍጫ'
        ]
        location_keywords = ['መገናኛ', 'ደራርቱ', 'መሰረት', 'ደፋር', 'ሞል', 'አዲስ', 'አበባ', 'ቢሮ', 'S05S06']
        price_pattern = r'(ዋጋ፦|በ)?\s*\d+\s*(ብር|birr|Birr|ETB)'
        product_modifier_keywords = ['ለሕፃን', 'የልጆች', 'የልብስ', 'የእጅ', 'የሚሰራ']

        # Label Products
        i = 0
        while i < len(tokens):
            if tokens[i] in product_keywords or tokens[i] in product_modifier_keywords:
                labels[i] = 'B-Product'
                # Look ahead to label subsequent tokens as I-Product
                j = i + 1
                while j < len(tokens) and (tokens[j] in product_keywords or tokens[j] in product_modifier_keywords or (tokens[j] not in location_keywords and not re.match(r'\d+', tokens[j]))):
                    labels[j] = 'I-Product'
                    j += 1
                i = j  # Skip processed tokens
            else:
                i += 1

        # Label Locations
        i = 0
        while i < len(tokens):
            if tokens[i] in location_keywords:
                labels[i] = 'B-LOC'
                # Look ahead to label subsequent tokens as I-LOC
                j = i + 1
                while j < len(tokens) and tokens[j] in location_keywords:
                    labels[j] = 'I-LOC'
                    j += 1
                i = j  # Skip processed tokens
            else:
                i += 1

        # Label Prices
        price_matches = re.finditer(price_pattern, message)
        for match in price_matches:
            price_text = match.group(0)
            price_tokens = word_tokenize(price_text)
            price_start_idx = 0
            for token in price_tokens:
                try:
                    idx = tokens.index(token, price_start_idx)
                    if token.isdigit():
                        labels[idx] = 'B-PRICE'
                    elif token in ['ብር', 'birr', 'Birr', 'ETB', 'ዋጋ፦', 'በ']:
                        labels[idx] = 'I-PRICE'
                    price_start_idx = idx + 1
                except ValueError:
                    logger.warning(f"Token '{token}' from price pattern not found in message")
                    continue

        logger.debug(f"Labeled tokens: {list(zip(tokens[:10], labels[:10]))}...")
        return tokens, labels
    except Exception as e:
        logger.error(f"Error labeling message: {message[:50]}...: {str(e)}")
        return [], []
    
# Convert to CoNLL format and save to file
def save_conll(messages, output_file, max_messages=50):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, message in enumerate(messages[:max_messages]):
                try:
                    logger.info(f"Processing message {idx + 1}/{min(len(messages), max_messages)}")
                    tokens, labels = label_entities(message)
                    if not tokens:
                        logger.warning(f"Skipping empty or failed message {idx + 1}")
                        continue
                    for token, label in zip(tokens, labels):
                        f.write(f"{token} {label}\n")
                    f.write("\n")  # Blank line to separate messages
                except Exception as e:
                    logger.error(f"Error processing message {idx + 1}: {str(e)}")
                    continue
        logger.info(f"Labeled dataset saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving CoNLL file to {output_file}: {str(e)}")
        raise

# Main function
def main():
    # Path to the provided dataset
    dataset_file = "../data/cleaned_message.csv"
    output_file = "../conLL/amharic_ner.conll"

    # Load messages
    messages = load_dataset(dataset_file)

    # Label and save up to 50 messages
    save_conll(messages, output_file, max_messages=50)
    logger.info("Labeling process completed successfully")

if __name__ == "__main__":
    main()