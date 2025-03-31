# rag/embeddings/clinical_bert.py

from transformers import AutoTokenizer, AutoModel

class ClinicalBERTEmbedder:
    def __init__(self):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def embed_text(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Get embeddings
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        
        return embeddings