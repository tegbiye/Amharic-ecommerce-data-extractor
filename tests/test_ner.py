import unittest
import pandas as pd
import os
import logging
from unittest.mock import patch
from io import StringIO
import re
from nltk.tokenize import word_tokenize
import nltk

# Import the functions to test
from src.labeling.label_conll_amharic import load_dataset, label_entities, save_conll

# Set up logging for tests
logger = logging.getLogger('src.labeling.label_conll_amharic')  # Match the logger name from ner_script.py
logger.setLevel(logging.INFO)
log_stream = StringIO()  # Capture logs in memory
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

class TestNERFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock NLTK download to avoid actual download
        cls.nltk_download_patcher = patch('nltk.download')
        cls.mock_nltk_download = cls.nltk_download_patcher.start()
        cls.mock_nltk_download.return_value = True

        # Create a temporary test CSV file
        cls.test_csv = 'test_cleaned_message.csv'
        test_data = """Message
"መገናኛ ሞል ውስጥ መዓዛን የሚሰጥ ማሽን ዋጋ፦ 500 ብር"
"መልዕክት የለም"
""
"ደራርቱ ለሕፃን ልብስ የሚስብ ክሬም በ 1000 ብር"
"አዲስ አበባ ውስጥ ፓወር ባንክ ዋጋ፦ 2000 ETB"
"""
        with open(cls.test_csv, 'w', encoding='utf-8') as f:
            f.write(test_data)

    @classmethod
    def tearDownClass(cls):
        # Stop mocking NLTK download
        cls.nltk_download_patcher.stop()
        # Clean up test CSV file
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)
        # Remove the stream handler to prevent memory leaks
        logger.removeHandler(stream_handler)

    def setUp(self):
        # Clear the log stream before each test
        log_stream.truncate(0)
        log_stream.seek(0)
        # Clear the log file if it exists
        self.log_file = 'labeling_log.log'
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_load_dataset_success(self):
        # Test loading and filtering of dataset
        messages = load_dataset(self.test_csv)
        self.assertEqual(len(messages), 3, "Should filter out 'መልዕክት የለም' and empty messages")
        self.assertIn("መገናኛ ሞል ውስጥ መዓዛን የሚሰጥ ማሽን ዋጋ፦ 500 ብር", messages)
        self.assertNotIn("መልዕክት የለም", messages)

    def test_load_dataset_file_not_found(self):
        # Test error handling for non-existent file
        with self.assertRaises(Exception):
            load_dataset("non_existent_file.csv")
        log_content = log_stream.getvalue()
        self.assertIn("Error loading dataset", log_content)

    def test_label_entities_valid_message(self):
        # Test entity labeling for a valid message
        message = "መገናኛ ሞል ውስጥ መዓዛን የሚሰጥ ማሽን ዋጋ፦ 500 ብር"
        tokens, labels = label_entities(message)
        expected_tokens = ['መገናኛ', 'ሞል', 'ውስጥ', 'መዓዛን', 'የሚሰጥ', 'ማሽን', 'ዋጋ፦', '500', 'ብር']
        expected_labels = ['B-LOC', 'I-LOC', 'O', 'B-Product', 'I-Product', 'I-Product', 'I-PRICE', 'B-PRICE', 'I-PRICE']
        self.assertEqual(tokens, expected_tokens, "Tokens should match expected")
        self.assertEqual(labels, expected_labels, "Labels should match expected")

    def test_label_entities_empty_message(self):
        # Test labeling an empty message
        tokens, labels = label_entities("")
        self.assertEqual(tokens, [], "Empty message should return empty tokens")
        self.assertEqual(labels, [], "Empty message should return empty labels")
        log_content = log_stream.getvalue()
        self.assertIn("Error labeling message", log_content)

    def test_label_entities_no_entities(self):
        # Test message with no recognizable entities
        message = "ይህ መልዕክት ምንም ነገር የለውም"
        tokens, labels = label_entities(message)
        expected_tokens = word_tokenize(message)
        expected_labels = ['O'] * len(expected_tokens)
        self.assertEqual(tokens, expected_tokens, "Tokens should match tokenized message")
        self.assertEqual(labels, expected_labels, "All tokens should be labeled 'O'")

    def test_save_conll_success(self):
        # Test saving to CoNLL format
        messages = [
            "መገናኛ ሞል ውስጥ መዓዛን የሚሰጥ ማሽን ዋጋ፦ 500 ብር",
            "ደራርቱ ለሕፃን ልብስ የሚስብ ክሬም በ 1000 ብር"
        ]
        output_file = 'test_amharic_ner.conll'
        save_conll(messages, output_file, max_messages=2)
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        expected_content = (
            "መገናኛ B-LOC\nሞል I-LOC\nውስጥ O\nመዓዛን B-Product\nየሚሰጥ I-Product\nማሽን I-Product\nዋጋ፦ I-PRICE\n500 B-PRICE\nብር I-PRICE\n\n"
            "ደራርቱ B-LOC\nለሕፃን B-Product\nልብስ I-Product\nየሚስብ I-Product\nክሬም I-Product\nበ I-PRICE\n1000 B-PRICE\nብር I-PRICE\n\n"
        )
        self.assertEqual(content, expected_content, "CoNLL output should match expected format")
        if os.path.exists(output_file):
            os.remove(output_file)

    def test_save_conll_empty_messages(self):
        # Test saving empty message list
        output_file = 'test_empty.conll'
        save_conll([], output_file, max_messages=2)
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, "", "Empty message list should produce empty file")
        if os.path.exists(output_file):
            os.remove(output_file)

    def test_save_conll_file_error(self):
        # Test error handling for unwritable file path
        messages = ["መገናኛ ሞል ውስጥ መዓዛን የሚሰጥ ማሽን ዋጋ፦ 500 ብር"]
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with self.assertRaises(Exception):
                save_conll(messages, "/invalid_path/output.conll")
        log_content = log_stream.getvalue()
        self.assertIn("Error saving CoNLL file", log_content)

if __name__ == '__main__':
    unittest.main()