import unittest
import os
import logging
from unittest.mock import patch, MagicMock
from io import StringIO
import numpy as np
from datasets import Dataset
import torch
from seqeval.metrics import classification_report
from src.ner_model.finetune_ner_model import load_conll, tokenize_and_align_labels, compute_metrics
from transformers.trainer_utils import EvalPrediction

# Set up logging for tests
logger = logging.getLogger('src.ner_model.finetune_ner_model')  # Match logger name from finetune_ner_model.py
logger.setLevel(logging.INFO)
log_stream = StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

class TestNERFinetuneFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary test CoNLL file
        cls.test_conll = 'test_amharic_ner.conll'
        test_data = """መገናኛ B-LOC
ሞል I-LOC
ውስጥ O
መዓዛን B-Product
የሚሰጥ I-Product
ማሽን I-Product
ዋጋ፦ I-PRICE
500 B-PRICE
ብር I-PRICE

ደራርቱ B-LOC
ለሕፃን B-Product
ልብስ I-Product
የሚስብ I-Product
ክሬም I-Product
በ I-PRICE
1000 B-PRICE
ብር I-PRICE
"""
        with open(cls.test_conll, 'w', encoding='utf-8') as f:
            f.write(test_data)

        # Mock tokenizer to return a BatchEncoding-like object
        cls.mock_tokenizer = MagicMock()
        cls.mock_batch_encoding = MagicMock()
        # Use a dictionary to store key-value pairs
        cls.mock_data = {
            'input_ids': torch.tensor([[101, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        }
        cls.mock_batch_encoding.__getitem__.side_effect = lambda key: cls.mock_data.get(key)
        cls.mock_batch_encoding.__setitem__.side_effect = lambda key, value: cls.mock_data.update({key: value})
        # Define word_ids as a function to handle multiple calls
        def word_ids(batch_index):
            word_id_lists = [
                [None, 0, 1, 2, 3, 4, 5, 6, 7, None],  # First sentence
                [None, 0, 1, 2, 3, 4, 5, 6, None]      # Second sentence
            ]
            return word_id_lists[batch_index]
        cls.mock_batch_encoding.word_ids = MagicMock(side_effect=word_ids)
        cls.mock_tokenizer.return_value = cls.mock_batch_encoding

        # Define label mappings
        cls.label2id = {
            "O": 0,
            "B-Product": 1,
            "I-Product": 2,
            "B-PRICE": 3,
            "I-PRICE": 4,
            "B-LOC": 5,
            "I-LOC": 6
        }
        cls.id2label = {v: k for k, v in cls.label2id.items()}

    @classmethod
    def tearDownClass(cls):
        # Clean up test CoNLL file
        if os.path.exists(cls.test_conll):
            os.remove(cls.test_conll)
        # Remove the stream handler
        logger.removeHandler(stream_handler)

    def setUp(self):
        # Clear the log stream before each test
        log_stream.truncate(0)
        log_stream.seek(0)
        # Clear the mock data dictionary before each test
        self.mock_data = {
            'input_ids': torch.tensor([[101, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        }
        # Clear the log file if it exists
        self.log_file = 'finetune_ner_log.log'
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_load_conll_success(self):
        # Test loading a valid CoNLL file
        data = load_conll(self.test_conll)
        expected_sentences = [
            ['መገናኛ', 'ሞል', 'ውስጥ', 'መዓዛን', 'የሚሰጥ', 'ማሽን', 'ዋጋ፦', '500', 'ብር'],
            ['ደራርቱ', 'ለሕፃን', 'ልብስ', 'የሚስብ', 'ክሬም', 'በ', '1000', 'ብር']
        ]
        expected_labels = [
            ['B-LOC', 'I-LOC', 'O', 'B-Product', 'I-Product', 'I-Product', 'I-PRICE', 'B-PRICE', 'I-PRICE'],
            ['B-LOC', 'B-Product', 'I-Product', 'I-Product', 'I-Product', 'I-PRICE', 'B-PRICE', 'I-PRICE']
        ]
        self.assertEqual(data['tokens'], expected_sentences, "Tokens should match expected")
        self.assertEqual(data['ner_tags'], expected_labels, "Labels should match expected")
        log_content = log_stream.getvalue()
        self.assertIn("Loaded 2 sentences", log_content)
        self.assertIn("Unique labels in dataset", log_content)

    def test_load_conll_file_not_found(self):
        # Test error handling for non-existent file
        with self.assertRaises(Exception):
            load_conll("non_existent_file.conll")
        log_content = log_stream.getvalue()
        self.assertIn("Error loading CoNLL file", log_content)

    def test_load_conll_malformed_line(self):
        # Test handling of malformed line
        malformed_conll = 'test_malformed.conll'
        with open(malformed_conll, 'w', encoding='utf-8') as f:
            f.write("መገናኛ B-LOC\nሞል I-LOC\nmalformed_line\nውስጥ O\n")
        data = load_conll(malformed_conll)
        self.assertEqual(len(data['tokens']), 1, "Should load one valid sentence")
        self.assertEqual(data['tokens'][0], ['መገናኛ', 'ሞል', 'ውስጥ'], "Should skip malformed line")
        log_content = log_stream.getvalue()
        self.assertIn("Skipping malformed line", log_content)
        if os.path.exists(malformed_conll):
            os.remove(malformed_conll)

    @patch('src.ner_model.finetune_ner_model.AutoTokenizer.from_pretrained')
    def test_tokenize_and_align_labels_success(self, mock_tokenizer):
        # Test tokenization and label alignment
        mock_tokenizer.from_pretrained.return_value = self.mock_tokenizer
        examples = {
            'tokens': [
                ['መገናኛ', 'ሞል', 'ውስጥ', 'መዓዛን', 'የሚሰጥ', 'ማሽን', 'ዋጋ፦', '500', 'ብር'],
                ['ደራርቱ', 'ለሕፃን', 'ልብስ', 'የሚስብ', 'ክሬም', 'በ', '1000', 'ብር']
            ],
            'ner_tags': [
                ['B-LOC', 'I-LOC', 'O', 'B-Product', 'I-Product', 'I-Product', 'I-PRICE', 'B-PRICE', 'I-PRICE'],
                ['B-LOC', 'B-Product', 'I-Product', 'I-Product', 'I-Product', 'I-PRICE', 'B-PRICE', 'I-PRICE']
            ]
        }
        dataset = Dataset.from_dict(examples)
        tokenized = tokenize_and_align_labels(dataset, self.mock_tokenizer, self.label2id)
        expected_labels = [
            [-100, 5, 6, 0, 1, 2, 2, 4, 3, -100],  # First sentence
            [-100, 5, 1, 2, 2, 2, 4, 3, -100]      # Second sentence
        ]
        self.assertEqual(tokenized['labels'], expected_labels, "Aligned labels should match expected")
        log_content = log_stream.getvalue()
        self.assertIn("Tokenizing batch of 2 sentences", log_content)
        self.assertIn("Aligned labels for batch: 2 sequences", log_content)

    @patch('src.ner_model.finetune_ner_model.AutoTokenizer.from_pretrained')
    def test_tokenize_and_align_labels_invalid_label(self, mock_tokenizer):
        # Test error handling for invalid label
        mock_tokenizer.from_pretrained.return_value = self.mock_tokenizer
        examples = {
            'tokens': [['መገናኛ', 'ሞል']],
            'ner_tags': [['B-LOC', 'INVALID']]
        }
        dataset = Dataset.from_dict(examples)
        with self.assertRaises(KeyError):
            tokenize_and_align_labels(dataset, self.mock_tokenizer, self.label2id)
        log_content = log_stream.getvalue()
        self.assertIn("Label 'INVALID' not found in label2id", log_content)

    def test_compute_metrics_success(self):
        # Test compute_metrics with mocked predictions and labels
        predictions = np.array([
            [[0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.5]],  # Predict B-LOC, I-LOC
            [[0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.0], [0.1, 0.1, 0.5, 0.1, 0.1, 0.0, 0.1]]   # Predict B-Product, I-Product
        ])
        labels = np.array([
            [5, 6],  # B-LOC, I-LOC
            [1, 2]   # B-Product, I-Product
        ])
        with patch('src.ner_model.finetune_ner_model.classification_report') as mock_report:
            mock_report.return_value = {
                'weighted avg': {'precision': 0.9, 'recall': 0.85, 'f1-score': 0.875}
            }
            metrics_fn = compute_metrics(self.id2label)
            result = metrics_fn(EvalPrediction(predictions, labels))
            self.assertEqual(result['precision'], 0.9, "Precision should match mock")
            self.assertEqual(result['recall'], 0.85, "Recall should match mock")
            self.assertEqual(result['f1'], 0.875, "F1 should match mock")
            log_content = log_stream.getvalue()
            self.assertIn("Evaluation metrics", log_content)

    def test_compute_metrics_error(self):
        # Test error handling in compute_metrics
        with patch('src.ner_model.finetune_ner_model.classification_report', side_effect=Exception("Metric error")):
            metrics_fn = compute_metrics(self.id2label)
            result = metrics_fn(EvalPrediction(np.array([[0, 0]]), np.array([[0, 0]])))
            self.assertEqual(result, {}, "Should return empty dict on error")
            log_content = log_stream.getvalue()
            self.assertIn("Error computing metrics", log_content)

if __name__ == '__main__':
    unittest.main()