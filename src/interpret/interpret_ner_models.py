import pandas as pd
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
import numpy as np
import shap
import lime.lime_text
from nltk.tokenize import word_tokenize
import nltk
from seqeval.metrics import classification_report

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Keep DEBUG for detailed SHAP logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interpret_ner_log.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK tokenizer
try:
    nltk.download('punkt', quiet=True)
    logger.info("NLTK punkt tokenizer downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK punkt tokenizer: {str(e)}")
    raise

# Load CoNLL file
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
        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        logger.info(f"Unique labels: {unique_labels}")
        return Dataset.from_dict({'tokens': sentences, 'ner_tags': labels})
    except Exception as e:
        logger.error(f"Error loading CoNLL file {file_path}: {str(e)}")
        raise

# Load new messages
def load_unlabeled_messages(file_path, skip_rows=50, max_messages=50):
    try:
        df = pd.read_csv(file_path)
        messages = df['Message'].dropna().tolist()
        messages = [msg for msg in messages if msg != "መልዕክት የለም"][skip_rows:skip_rows+max_messages]
        logger.info(f"Loaded {len(messages)} new messages from {file_path}")
        return messages
    except Exception as e:
        logger.error(f"Error loading messages from {file_path}: {str(e)}")
        raise

# Predict entities and get logits
def predict_entities(message, tokenizer, model, id2label, device):
    try:
        tokens = word_tokenize(message)
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=512
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        predictions = np.argmax(logits, axis=1)
        word_ids = inputs.word_ids()
        aligned_labels = []
        aligned_logits = []
        prev_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                aligned_labels.append(id2label[predictions[i]])
                aligned_logits.append(logits[i])
            prev_word_idx = word_idx
        return tokens, aligned_labels, aligned_logits
    except Exception as e:
        logger.error(f"Error predicting entities for message: {message[:50]}...: {str(e)}")
        return word_tokenize(message), [], []

# SHAP explainer
def apply_shap(message, model, tokenizer, device):
    try:
        class ModelLogitsWrapper:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device

            def __call__(self, *args, **kwargs):
                # Handle inputs from SHAP's masker (expecting raw text strings)
                inputs = args[0] if args else kwargs.get('inputs', [])
                logger.debug(f"Received inputs type: {type(inputs)}, shape/content: {inputs}")

                # Convert inputs to list of strings, handling NumPy arrays or lists
                if isinstance(inputs, np.ndarray):
                    inputs = inputs.tolist()
                if not isinstance(inputs, list):
                    inputs = [inputs]
                inputs = [str(x).strip() for x in inputs if str(x).strip()]

                # Handle empty or invalid inputs
                if not inputs:
                    logger.debug("Empty or invalid text inputs received")
                    return torch.zeros((1, len(self.model.config.id2label)), device=self.device)

                # Tokenize raw text
                try:
                    encoded = self.tokenizer(
                        inputs,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        max_length=512
                    )
                    input_ids = encoded['input_ids'].to(self.device)
                    attention_mask = encoded['attention_mask'].to(self.device)
                except Exception as e:
                    logger.debug(f"Tokenization error: {str(e)}")
                    return torch.zeros((len(inputs), len(self.model.config.id2label)), device=self.device)

                # Validate tokenized inputs
                if input_ids.shape[0] == 0 or input_ids.shape[1] == 0 or torch.all(input_ids == self.tokenizer.pad_token_id):
                    logger.debug(f"Invalid tokenized inputs: shape={input_ids.shape}")
                    return torch.zeros((input_ids.shape[0], len(self.model.config.id2label)), device=self.device)

                # Run model
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Aggregate token-level logits to sentence-level (mean across tokens)
                if logits.ndim == 3:
                    logits = torch.mean(logits, dim=1)  # Shape: (batch_size, num_labels)
                logger.debug(f"Logits shape: {logits.shape}")
                return logits

        # Initialize explainer with tokenizer as masker
        explainer = shap.Explainer(ModelLogitsWrapper(model, tokenizer, device), tokenizer)
        shap_values = explainer([message], max_evals=100)
        logger.info(f"Computed SHAP values for message: {message[:50]}...")
        return str(shap_values.values)
    except Exception as e:
        logger.error(f"Error in SHAP explanation for message: {message[:50]}...: {str(e)}")
        logger.debug(f"Traceback: ", exc_info=True)
        return "SHAP_ERROR"

# LIME explainer
def apply_lime(message, tokenizer, model, id2label, device):
    try:
        def model_predict_proba_lime(texts):
            all_probs = []
            for text in texts:
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=512
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits.cpu().numpy()
                token_logits = logits[0]
                token_probs = torch.softmax(torch.tensor(token_logits), dim=-1).numpy()
                sentence_probs = np.max(token_probs, axis=0)
                all_probs.append(sentence_probs)
            return np.array(all_probs)

        explainer = lime.lime_text.LimeTextExplainer(class_names=list(id2label.values()))
        explanation = explainer.explain_instance(
            message,
            model_predict_proba_lime,
            num_features=10,
            num_samples=200
        )
        logger.info(f"Computed LIME explanation for message: {message[:50]}...")
        return explanation.as_list()
    except Exception as e:
        logger.error(f"Error in LIME explanation for message: {message[:50]}...: {str(e)}")
        return []

# Main function
def main():
    try:
        # Define paths and model
        conll_file = 'conLL/amharic_ner.conll'
        messages_file = 'data/cleaned_message.csv'
        model_dir = 'models/amharic_ner_xlmr'
        output_file = 'data/interpretability_results.csv'
        report_file = 'Task5_Interpretability_Report.md'

        # Label mappings
        label2id = {
            "O": 0, "B-Product": 1, "I-Product": 2, "B-PRICE": 3,
            "I-PRICE": 4, "B-LOC": 5, "I-LOC": 6
        }
        id2label = {v: k for k, v in label2id.items()}
        logger.info(f"Label mappings: {label2id}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Loaded model and tokenizer from {model_dir}, using device: {device}")

        # Load datasets
        dataset = load_conll(conll_file)
        val_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))
        messages = load_unlabeled_messages(messages_file)
        logger.info(f"Validation set size: {len(val_dataset)}")

        # Analyze validation set and new messages
        results = []
        difficult_cases = []
        for idx, example in enumerate(val_dataset):
            message = " ".join(example['tokens'])
            true_labels = example['ner_tags']
            tokens, pred_labels, logits = predict_entities(message, tokenizer, model, id2label, device)
            
            max_logits = np.max(logits, axis=1) if len(logits) > 0 and len(logits[0]) > 0 else np.array([])
            low_confidence = np.any(max_logits < 0.9) if max_logits.size > 0 else False
            
            mismatch = len(pred_labels) != len(true_labels) or any(p != t for p, t in zip(pred_labels, true_labels))

            if mismatch or low_confidence:
                difficult_cases.append({
                    "message": message, "true_labels": true_labels, "pred_labels": pred_labels,
                    "logits": max_logits.tolist() if max_logits.size > 0 else []
                })

            if idx < 5:
                shap_values = apply_shap(message, model, tokenizer, device)
                lime_weights = apply_lime(message, tokenizer, model, id2label, device)
            else:
                shap_values = "SKIPPED"
                lime_weights = "SKIPPED"

            results.append({
                "message": message, "true_labels": ", ".join(true_labels),
                "pred_labels": ", ".join(pred_labels), "shap_values": shap_values,
                "lime_weights": str(lime_weights)
            })
            logger.info(f"Processed validation example {idx + 1}/{len(val_dataset)}")

        # Analyze new messages
        for idx, message in enumerate(messages):
            tokens, pred_labels, logits = predict_entities(message, tokenizer, model, id2label, device)
            
            if idx < 5:
                shap_values = apply_shap(message, model, tokenizer, device)
                lime_weights = apply_lime(message, tokenizer, model, id2label, device)
            else:
                shap_values = "SKIPPED"
                lime_weights = "SKIPPED"

            results.append({
                "message": message, "true_labels": "N/A", "pred_labels": ", ".join(pred_labels),
                "shap_values": shap_values, "lime_weights": str(lime_weights)
            })
            logger.info(f"Processed new message {idx + 1}/{len(messages)}")

        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved interpretability results to {output_file}")

        # Generate interpretability report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Named Entity Recognition Model Interpretability Report\n\n")
            f.write("This report provides insights into the behavior of the Amharic NER model using SHAP and LIME.\n\n")
            
            f.write("## 1. Model Performance\n")
            f.write("The fine-tuned `Davlan/afro-xlmr-base` model achieved an F1 score of 0.9187 (from Task 4) on the Amharic NER dataset.\n\n")

            f.write("## 2. Difficult Cases from Validation Set\n\n")
            if difficult_cases:
                f.write(f"Identified {len(difficult_cases)} difficult cases (mismatches or low-confidence predictions, logits < 0.9):\n\n")
                for i, case in enumerate(difficult_cases[:5]):
                    f.write(f"### Case {i+1}:\n")
                    f.write(f"- **Message:** {case['message']}\n")
                    f.write(f"- **True Labels:** {', '.join(case['true_labels'])}\n")
                    f.write(f"- **Predicted Labels:** {', '.join(case['pred_labels'])}\n")
                    f.write(f"- **Max Logits (Confidence):** {', '.join(f'{l:.2f}' for l in case['logits'])}\n\n")
            else:
                f.write("No difficult cases identified based on the defined criteria.\n\n")

            f.write("## 3. SHAP and LIME Explanations\n\n")
            f.write("SHAP and LIME were applied to the first 5 validation examples and new messages to explain token contributions.\n\n")
            explained_results = [r for r in results if r.get('shap_values') and r.get('shap_values') != 'SHAP_ERROR' and r.get('shap_values') != 'SKIPPED']
            if explained_results:
                for i, row in enumerate(explained_results[:5]):
                    f.write(f"### Explanation for Message {i+1}:\n")
                    f.write(f"- **Message:** {row['message']}\n")
                    f.write(f"- **Predicted Labels:** {row['pred_labels']}\n")
                    f.write(f"- **SHAP Values:** {row['shap_values']}\n")
                    f.write(f"- **LIME Weights:** {row['lime_weights']}\n\n")
            else:
                f.write("No SHAP/LIME explanations generated (possible errors during processing).\n\n")
            
            f.write("## 4. Recommendations\n")
            f.write("- **Increase Dataset Size**: Label 300–500 additional messages to improve robustness, especially for rare labels like `I-LOC`.\n")
            f.write("- **Handle Ambiguity**: Add training examples with ambiguous tokens (e.g., numbers like '360' as attributes or prices).\n")
            f.write("- **Optimize Inference**: Quantize the model to reduce inference time (0.5105s/sample from Task 4).\n")
            f.write("- **Custom Explainers**: Develop NER-specific explainers for better token-level insights.\n\n")

            f.write("## 5. Conclusion\n")
            f.write("SHAP and LIME provide valuable insights into the `Davlan/afro-xlmr-base` model’s decisions for Amharic NER. Addressing dataset limitations and ambiguities will further enhance performance. Results are saved in `interpretability_results.csv`, and logs are in `interpret_ner_log.log`.\n")

        logger.info(f"Generated interpretability report to {report_file}")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()