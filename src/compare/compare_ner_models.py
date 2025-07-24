import pandas as pd
import logging
import time
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
        logging.FileHandler('compare_ner_log.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log environment details
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
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
        
        unique_labels = set(label for sent_labels in labels for label in sent_labels)
        label_counts = {label: sum(sent_labels.count(label) for sent_labels in labels) for label in unique_labels}
        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        logger.info(f"Unique labels: {unique_labels}")
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
                    aligned_labels.append(-100)
                elif word_idx != prev_word_idx:
                    try:
                        aligned_labels.append(label2id[label[word_idx]])
                    except KeyError as e:
                        logger.error(f"Label '{label[word_idx]}' not found in label2id: {label2id}")
                        raise
                else:
                    aligned_labels.append(-100)
                prev_word_idx = word_idx
            labels.append(aligned_labels)
        
        tokenized_inputs["labels"] = labels
        logger.debug(f"Aligned labels for batch: {len(labels)} sequences")
        return tokenized_inputs
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        raise

# Compute metrics
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

# Fine-tune and evaluate a single model
def fine_tune_model(model_name, dataset, label2id, id2label, output_dir):
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Loaded {model_name} on {device}")

        # Split dataset
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")

        # Tokenize datasets
        tokenized_train = train_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True
        )
        tokenized_val = val_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True
        )
        logger.info(f"Tokenized datasets for {model_name}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir=f'./logs_{model_name.split("/")[-1]}',
            logging_steps=10,
            load_best_model_at_end=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics(id2label)
        )

        # Measure training time
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        logger.info(f"Training completed for {model_name} in {training_time:.2f} seconds")

        # Evaluate on validation set
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results for {model_name}: {eval_results}")

        # Measure inference time
        start_time = time.time()
        for i in range(len(val_dataset)):
            inputs = tokenizer(
                val_dataset[i]['tokens'],
                is_split_into_words=True,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                model(**inputs)
        inference_time = (time.time() - start_time) / len(val_dataset)
        logger.info(f"Average inference time per sample for {model_name}: {inference_time:.4f} seconds")

        # Save model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved {model_name} to {output_dir}")

        return {
            "model_name": model_name,
            "precision": eval_results.get("eval_precision", 0.0),
            "recall": eval_results.get("eval_recall", 0.0),
            "f1": eval_results.get("eval_f1", 0.0),
            "training_time": training_time,
            "inference_time": inference_time
        }
    except Exception as e:
        logger.error(f"Error fine-tuning {model_name}: {str(e)}")
        return {"model_name": model_name, "precision": 0.0, "recall": 0.0, "f1": 0.0, "training_time": 0.0, "inference_time": 0.0}

# Main function
def main():
    try:
        # Define dataset and models
        dataset_file = 'amharic_ner.conll'
        models = [
            ("Davlan/afro-xlmr-base", "./amharic_ner_xlmr"),
            ("google-bert/bert-base-multilingual-cased", "./amharic_ner_mbert"),
            ("distilbert/distilbert-base-multilingual-cased", "./amharic_ner_distilbert")
        ]
        output_comparison_file = "model_comparison.csv"

        # Define label mappings
        label2id = {
            "O": 0,
            "B-Product": 1,
            "I-Product": 2,
            "B-PRICE": 3,
            "I-PRICE": 4,
            "B-LOC": 5,
            "I-LOC": 6
        }
        id2label = {v: k for k, v in label2id.items()}
        logger.info(f"Label mappings: {label2id}")

        # Load dataset
        data = load_conll(dataset_file)
        dataset = Dataset.from_dict(data)

        # Verify label consistency
        unique_labels = set(label for sent_labels in data['ner_tags'] for label in sent_labels)
        missing_labels = unique_labels - set(label2id.keys())
        if missing_labels:
            logger.error(f"Labels in dataset not in label2id: {missing_labels}")
            raise ValueError(f"Labels in dataset not in label2id: {missing_labels}")

        # Fine-tune and evaluate each model
        results = []
        for model_name, output_dir in models:
            logger.info(f"Starting fine-tuning for {model_name}")
            result = fine_tune_model(model_name, dataset, label2id, id2label, output_dir)
            results.append(result)

        # Save comparison results
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_comparison_file, index=False, encoding='utf-8')
        logger.info(f"Saved model comparison to {output_comparison_file}")

        # Select best model (highest F1-score, with speed as tiebreaker)
        best_model = max(results, key=lambda x: (x["f1"], -x["inference_time"]))
        logger.info(f"Best model: {best_model['model_name']} (F1: {best_model['f1']:.4f}, Inference Time: {best_model['inference_time']:.4f}s)")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()