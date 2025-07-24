import pandas as pd
from datetime import datetime, timedelta
import re
from transformers import pipeline
import numpy as np
import random

# Simulate NER pipeline (replace with actual Davlan/afro-xlmr-base model)
def extract_entities(message):
    # Simulated NER: Extract price and product
    price_match = re.search(r'ዋጋ፦\s*(\d+)(?:\s*ብር)?', message)
    price = int(price_match.group(1)) if price_match else None
    
    # Extract product (text before ዋጋ፦, simplified)
    product = message.split("ዋጋ፦")[0].strip() if "ዋጋ፦" in message else None
    if product and any(x in product.lower() for x in ["pcs", "ፓርቲሽን"]):
        product = None  # Handle ambiguous products like "4PCS"
    
    return price, product

