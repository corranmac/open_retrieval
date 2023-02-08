import argparse
from pathlib import Path
import faiss, glob, time, os, json
import torch
import pandas as pd
import numpy as np

from src.data_loader import DatasetIndexLoader
from src.model import Retriever


#Retrieve k most similar matches to query
def retrieve(retriever,data,query="who was the last man on the moon?",device='cpu',column=None,extra_columns=None,k=5,use_nn=False,nn_threshold=0.7,json_path=None):
  
  index = data.index
  dataset = data.dataset
  tokenizer = retriever.tokenizer
  model = retriever.model

  #Set device
  if not device == 'cpu':
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
  else:
    device = 'cpu'

  #Validate nn_threshold
  if use_nn:
    try: 
      nn_threshold = float(nn_threshold)
    except:
      raise SystemExit("The specified nn_threshold is not a valid float")

  #Tokenize query
  query_vector = retriever.embed_text([query])

  #Get indices for top k matches
  distances, indices = index.search(query_vector, k)

  results={}
  print("Retrieving top {} results".format(k))

  if dataset is not None:
    if use_nn:
      for i, (dist, indice) in enumerate(zip(distances[0], indices[0])): #Loop through k results
        text = str(dataset.iloc[indice][column] )
        extra_indices = []
        result = {'distance': dist, 'text':text, 'indice':indice}

        # get embedding of neighboring 3-sentence segments
        try:
          embeddings  = retriever.embed_text([str(dataset.iloc[indice-1][column]), str(dataset.iloc[indice][column]), str(dataset.iloc[indice+1][column]) ])

          if cos_sim_2d(embeddings[0].reshape(1, 768), embeddings[-1].reshape(1, 768)) > nn_threshold: 
            extra_indices.append(indice-1)
            text = str(dataset.iloc[indice-1][column]) +" "+ str(dataset.iloc[indice][column])

          if cos_sim_2d(embeddings[0].reshape(1, 768), embeddings[1].reshape(1, 768)) > nn_threshold:
            extra_indices.append(indice+1)
            text += str(dataset.iloc[indice+1][column])
          
          #Add extra indices
          result['context_text'] = text
          result['extra_indices'] = extra_indices

          #Add metadata from dataset
          if not extra_columns == None:
            for ecolumn in extra_columns:
              result[ecolumn]= str(dataset.iloc[indice][ecolumn])

        except: #If the similarity matching fails
          if not extra_columns == None: #Add extra metadata
            for ecolumn in extra_columns:
              result[ecolumn]= dataset.iloc[indice][ecolumn]

        #Add result data to results
        results[str(i)]= result

    else: # If use_nn is False
      for i, (dist, indice) in enumerate(zip(distances[0], indices[0])): #Loop through k results
        text = str(dataset.iloc[indice][column])
        result = {'distance': dist, 'text':text, 'indice':indice}
        if not extra_columns == None: #Add extra metadata
          for ecolumn in extra_columns:
            result[ecolumn]= dataset.iloc[indice][ecolumn]
        #Add result data to results
        results[str(i)]= result
    
  else: #If dataset == None
    for i, (dist, indice) in enumerate(zip(distances[0], indices[0])): #Loop through k results
      result = {'distance': dist, 'indice':indice}
      #Add result data to results
      results[str(i)]= result

  if k<20:
    print(results)

  #Writes results to json if specified
  if json_path is not None:
    for i in results.keys():
      for ii in results[i].keys():
        results[i][ii] = str(results[i][ii])
    with open(json_path, "w") as jsonfile:
      json.dump(results, jsonfile)
  
  return results

if __name__ == '__main__':
    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Who was the last man on the moon?", help="The query for retrieval", required=False)
    parser.add_argument("--index_path", help="The path to faiss index containing the items for retrieval", required=True)
    parser.add_argument("--device", default="cpu", help="Device to run proccessing, i.e. gpu or cpu", required=False)
    parser.add_argument("--retriever", default="facebook/contriever-msmarco", help="Transformers automodel to be used for embedding for retrieval", required=True)
    parser.add_argument("--dataset_path", default=None, help="The path to the parquet dataset containing the items for retrieval as strings", required=False)
    parser.add_argument("--column", default=None, help="Dataset column name containing the sentences/paragraphs for retrieval (str)", required=False)
    parser.add_argument("--extra_columns", default=None, help="Columns to return alongside the result indexs when using a dataset", required=False)
    parser.add_argument("--n", default=5, help="The number of results to retrieve", required=False)
    parser.add_argument("--use_nn", default=False, help="Determine 3 nearest neigbours, True or False", required=False)
    parser.add_argument("--nn_threshold", default=0.7, help="The minimum similarity for 3 nearest neighbours when using use_nn", required=False)
    parser.add_argument("--json_path", default="", help="Optionally output the results to a json file", required=False)
    args = parser.parse_args()

    #Parse arguments
    k = int(args.n)
    use_nn = args.use_nn
    nn_threshold = args.nn_threshold
    query = args.query
    column = args.column
    extra_columns = args.extra_columns
    device = args.device
    json_path = args.json_path
    dataset_path = args.dataset_path
    index_path= args.index_path
    retriever = args.retriever

    if dataset_path is not None:
      if column is None:
        raise SystemExit("Please provide the name of the column that includes the text for the embeddings in your dataset")

    #Load retriever and its tokenizer
    retriever = Retriever(model_name=retriever, device=device)

    #Load index and dataset
    data = DatasetIndexLoader(index=index_path,dataset=dataset_path, column=column)

    # Call the retrieval function
    results = retrieve(retriever,data,query,device,column)
