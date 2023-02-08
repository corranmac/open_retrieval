import faiss
import pandas as pd
from pathlib import Path

class DatasetIndexLoader:

    def load_dataset(self, dataset, column):
      # Attempt to load the dataset
      if dataset is not None:
        dataset = Path(dataset)
        if column is None:
          raise SystemExit("Please provide the name of the column that includes the text for the embeddings in your dataset")
        if not dataset.exists():
          raise SystemExit("The dataset specified doesn't exist")
        else:
          print("Loading dataset...")
          try: 
            dataset = pd.read_parquet(dataset, engine='fastparquet')
          except:
              raise SystemExit("The dataset is not a valid parquet")
      return dataset

    def load_index(self, index):
      #Attempt to read the faiss index
      index = Path(index)
      if not index.exists():
        raise SystemExit("The index specified doesn't exist")
      try: 
          print("Loading faiss index...")
          index = str(index)
          index = faiss.read_index(index, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
      except:
          raise SystemExit("The index is not valid")
      return index

    def __init__(self, index,dataset=None, column=None):
      self.index = self.load_index(index)
      self.dataset = self.load_dataset(dataset,column)
      self.column = column