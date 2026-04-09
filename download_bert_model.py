from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', cache_dir='./bert-base-uncased')

print("Downloaded BERT model and tokenizer locally to './bert-base-uncased'")
