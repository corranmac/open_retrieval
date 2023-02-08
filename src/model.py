import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class Retriever():
  def __init__(self, model_name="facebook/contriever-msmarco", device="cpu"):
    #Set device
    if not device == "cpu":
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    else:
      self.device = 'cpu'

    #Load tokenizer & retriever
    print("Loading retriever model...")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    self.model.to(self.device)
  
  def mean_pooling(self, token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

  def embed_text(self, text):
    inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
    outputs = self.model(**inputs)
    embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
    query_vector = np.asarray(embeddings.cpu().detach().numpy()).reshape(1, 768)

    return query_vector
  