#!/usr/bin/env python
"""Initialize BERT model to ensure all files are downloaded."""

import os
import sys

print("Downloading BERT model files...")

try:
    from transformers import BertTokenizer, BertModel
    
    # Download tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    print("✓ BERT model successfully downloaded and cached")
    sys.exit(0)
except Exception as e:
    print(f"✗ Error downloading BERT model: {e}")
    sys.exit(1)
