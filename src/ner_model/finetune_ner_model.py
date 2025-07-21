import pandas as pd
import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
import numpy as np
from seqeval.metrics import classification_report
import torch
import transformers

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetune_ner_log.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log transformers version and device
logger.info(f"Using transformers version: {transformers.__version__}")
logger.info(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Function to load CoNLL file
def load_conll(file_path):
    try:
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        token, label = line.split()
                        current_sentence.append(token)
                        current_labels.append(label)
                    except ValueError:
                        logger.warning(f"Skipping malformed line: {line}")
                        continue
                else:
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
            # Append the last sentence if it exists
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
        
        # Verify unique labels and their counts
        unique_labels = set(label for sent_labels in labels for label in sent_labels)
        label_counts = {label: sum(sent_labels.count(label) for sent_labels in labels) for label in unique_labels}
        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        logger.info(f"Unique labels in dataset: {unique_labels}")
        logger.info(f"Label counts: {label_counts}")
        return {'tokens': sentences, 'ner_tags': labels}
    except Exception as e:
        logger.error(f"Error loading CoNLL file {file_path}: {str(e)}")
        raise

# Tokenize and align labels
def tokenize_and_align_labels(examples, tokenizer, label2id):
    try:
        logger.debug(f"Tokenizing batch of {len(examples['tokens'])} sentences")
        tokenized_inputs = tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt"
        )

        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # Special tokens get -100
                elif word_idx != prev_word_idx:
                    try:
                        aligned_labels.append(label2id[label[word_idx]])
                    except KeyError as e:
                        logger.error(f"Label '{label[word_idx]}' not found in label2id: {label2id}")
                        raise
                else:
                    aligned_labels.append(-100)  # Subword tokens get -100
                prev_word_idx = word_idx
            labels.append(aligned_labels)
        
        tokenized_inputs["labels"] = labels
        logger.debug(f"Aligned labels for batch: {len(labels)} sequences")
        return tokenized_inputs
    except Exception as e:
        logger.error(f"Error in tokenization and label alignment: {str(e)}")
        raise

# Compute metrics for evaluation
def compute_metrics(id2label):
    def metrics(p: EvalPrediction):
        try:
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            
            true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
            pred_labels = [[id2label[p] for p, l in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
            
            report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            logger.info(f"Evaluation metrics: {report}")
            return {
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1": report["weighted avg"]["f1-score"]
            }
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {}
    return metrics

# Main function
def main():
    try:
        # Define paths and model
        dataset_file = '../conLL/amharic_ner.conll'
        model_name = "Davlan/afro-xlmr-base"
        output_dir = "../models/amharic_ner_model"
        
        # Define label mappings (matching Task 2 labels exactly)
        label2id = {
            "O": 0,
            "B-Product": 1,
            "I-Product": 2,
            "B-Price": 3,
            "I-Price": 4,
            "B-LOC": 5,
            "I-LOC": 6
        }
        id2label = {v: k for k, v in label2id.items()}
        logger.info(f"Label mappings: {label2id}")
        
        logger.info("Starting fine-tuning process")
        
        # Load dataset
        data = load_conll(dataset_file)
        
        # Verify label consistency
        unique_labels = set(label for sent_labels in data['ner_tags'] for label in sent_labels)
        missing_labels = unique_labels - set(label2id.keys())
        if missing_labels:
            logger.error(f"Labels in dataset not in label2id: {missing_labels}")
            raise ValueError(f"Labels in dataset not in label2id: {missing_labels}")
        
        dataset = Dataset.from_dict(data)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        logger.info(f"Loaded model and tokenizer: {model_name}")
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True
        )
        tokenized_val = val_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True
        )
        logger.info("Tokenized datasets successfully")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Reduced for small dataset
            per_device_eval_batch_size=8,
            num_train_epochs=3,  # Increased to improve training
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            no_cuda=not torch.cuda.is_available(),
            pin_memory=torch.cuda.is_available()
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics(id2label)
        )
        logger.info("Initialized trainer")
        
        # Train model
        trainer.train()
        logger.info("Training completed")
        
        # Save model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()