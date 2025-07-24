# Building an Amharic E-commerce Data Extractor

### Transform messy Telegram posts into a smart FinTech engine that reveals which vendors are the best candidates for a loan.

---

## Project Structure
Amharic-ecommerce-data-extractor
<pre>
|_____.amhenv/
|---- .github/
|     |--- workflows
|     |    |--- unittests.yml
|---- data/
|     |---- telegram_data.csv (raw)
|     |---- cleaned_message.csv (Cleaned)
|     |---- model_comparison.csv   # Model comparison
|     |---- interpretability_results.csv
|     |---- vendor_metrics.csv
|     |---- vendor_scorecard.md
|-----conLL/
|     |____ amharic_ner.conll
|-----models/
|     |----- amharic_ner_distilbert
|     |----- amharic_ner_mbert
|     |----- amharic_ner_xlmr
|-----logs/
|     |--- interpret_ner_log.log
|     |--- compare_ner_log.log
|     |--- finetune_ner_log.log
|     |--- labeling_log.log
|---- notebooks/
|     |--- README.md
|     |--- data-ingestion-preprocessing.ipynb
|     |--- finetune-amharic-ner.ipynb
|     |--- compare_ner_models.ipynb
|     |--- interpret_ner_model.ipynb
|     |--- vendor_score_baord.ipynb
|     |____ data-labeling-conLL.ipynb
|---- scripts/
|     |--- __init__.py
|     |--- data_loader.py
|     |--- preprocess.py
|     |--- telegram_scrapper.py (function for scraping telegram channel data)
|-----src/
|     |--- __init__.py
|     |____ labeling/
|           |____ label_conll_amharic.py  # labeling in conll format
|     |____ ner_model/
|           |____ finetune_ner_model.py   # finetuning model
|     |____ compare/
|           |____ compare_ner_models.py   # compare by finetuning
|     |____ vendor_board/
|           |____ vendor_analytics_board.py # vendor_board visualization
|---- tests/
|     |--- __init__.py
|     |--- test_sample.py
|     |--- test_finetune_ner.py
|     |____ test_ner.py
|---- .gitignore
|---- requirements.txt (Dependencies)
|---- LICENSE
|____ README.md
</pre>

## Task 1
   1. Data scraping
      Scraped data from seven channels namely:
      - Zemen Express®
      - NEVA COMPUTER®
      - HellooMarket
      - ሞደርን ሾፒንግ ሴንተር MODERN SHOPPING CENTER
      - qnash.com - ቅናሽ ®️
      - አዳማ ገበያ - Adama gebeya
      - Sheger online-store
      - Scraped around 37378 of raw data.
   2. Data Preprocessing
      - Checked for the missing values and found some significant amount of missing values fixed
      - made cleaning of the Amharic text as the task is for Amharic language where removed english text and other punctuations, emojis, tags.
      - Saved the cleaned message to be used for the task 2
## Task 2
Based on the task objective 30–50 messages from the "Message" column of the provided cleaned_message.csv dataset is used to create in CoNLL format for Named Entity Recognition (NER). The entities to be identified and labeled include:

   •	Product: Items being advertised 

   •	Price: Monetary values 

   •	Location: Place names or addresses 

   •	Labels follow the BIO scheme: B- (Beginning), I- (Inside), and O (Outside) for each entity type.

1.	Initial Script Development:

      o	Created a Python script (label_conll_amharic.py) to load the dataset, tokenize messages, label entities, and save the output in CoNLL format.

      o	Used NLTK's word_tokenize for tokenization, suitable for Amharic text.

      o	Defined keyword lists for products and locations and a regular expression pattern for prices

      o	Filtered out invalid messages and processed up to 50 messages.

2.	Dataset-Specific Updates:

      o	Updated the script to handle the provided cleaned_message.csv, which contains 37,377 rows, with the "Message" column in Amharic.

      o	Refined keyword lists based on the dataset's content for products, and for locations.

      o	Enhanced product detection with contextual keywords to capture multi-word entities.

3.	Logging Integration:

      o	Added logging using Python's logging module, outputting to both labeling_log.log and the console.

      o	Log levels:

         - INFO: Key steps (e.g., dataset loading, file saving, message processing).

         - DEBUG: Detailed tokenization and labeling output (first 10 tokens per message).

         - WARNING: Non-critical issues (e.g., unmatched price tokens).

         - ERROR: Critical failures (e.g., file loading errors).

      o	Included try-except blocks to log errors and ensure robustness.

4.	Output Generation:

      o	Processed up to 50 valid messages and saved the labeled data to amharic_ner.conll.

      o	Ensured CoNLL format

## Getting Started
1. Clone the Repository
   ``` 
   git clone https://github.com/tegbiye/Amharic-ecommerce-data-extractor.git
   
   ```
   ```
    cd Amharic-ecommerce-data-extractor
   ```
2. Create environment
   ```
   python -m venv .amhenv
   
   ```
   Windows
   ```
   .venv\Scripts\activate
   ```
   Linux/Mac
   ```
   source .venv\bin\activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with proper attribution.


