# Building an Amharic E-commerce Data Extractor

### Transform messy Telegram posts into a smart FinTech engine that reveals which vendors are the best candidates for a loan.

---

## Project Structure
Amharic-ecommerce-data-extractor
<pre>
|_____.env/
|---- .github/
|     |--- workflows
|     |    |--- unittests.yml
|---- data/
|     |---- telegram_data.csv (raw)
|     |---- cleaned_message.csv (Cleaned)
|---- notebooks/
|     |--- README.md
|     |--- data-ingestion-preprocessing.ipynb
|---- scripts/
|     |--- __init__.py
|     |--- data_loader.py
|     |--- preprocess.py
|     |--- telegram_scrapper.py (function for scraping telegram channel data)
|-----src/
|     |--- __init__.py
|---- tests/
|     |--- __init__.py
|     |--- test_sample.py
|---- .gitignore
|---- requirements.txt (Dependencies)
|---- LICENSE
|____ README.md
</pre>

## Task One
   1. Data scraping
      Scraped data from seven channels namely:
      - Zemen Express®
      - NEVA COMPUTER®
      - HellooMarket
      - ሞደርን ሾፒንግ ሴንተር MODERN SHOPPING CENTER
      - qnash.com - ቅናሽ ®️
      - አዳማ ገበያ - Adama gebeya
      - Sheger online-store
      Scraped around 37378 of raw data.
   2. Data Preprocessing
      Checked for the missing values and found some significant amount of missing values 
      fixed
      made cleaning of the Amharic text as the task is for Amharic language where removed english text and other punctuations, emojis, tags.
      Saved the cleaned message to be used for the task 2
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
   python -m venv .venv
   
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


