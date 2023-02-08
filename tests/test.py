import os
from pathlib import Path
import torch
from main import retrieve
from src.data_loader import DatasetIndexLoader
from src.model import Retriever

if not os.path.exists("wikipedia-3sentence-level-retrieval-index/knn.index"): #High Ram (20GB required) example
  !git clone https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index/

index_path = "wikipedia-3sentence-level-retrieval-index/knn.index"
dataset_path = "wikipedia-3sentence-level-retrieval-index/wikipedia-en-sentences.parquet"
retriever = "facebook/contriever-msmarco"
column = "text_snippet"
extra_columns = ["position_in_source_text"]

query = "Who was the last man on the moon?"
json_path = "results.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#Load retriever and its tokenizer
retriever = Retriever(model_name=retriever, device=device)

#Load index and dataset
data = DatasetIndexLoader(index=index_path,dataset=dataset_path, column=column)

# Call the retrieval function
results = retrieve(retriever,data,query,device,column,extra_columns="position_in_source_text")
