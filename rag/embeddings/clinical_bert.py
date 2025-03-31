# rag/embeddings/clinical_bert.py

from transformers import AutoTokenizer, AutoModel

class ClinicalBERTEmbedder:
    def __init__(self):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Use the embedding of the [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding