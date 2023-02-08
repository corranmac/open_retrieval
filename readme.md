# Open Retrieval

Retrieve semantically close text embeddings using a prebuilt FAISS index and retrieval model from HF transformers.

Query the faiss index and optionally retrieve metadata from a parquet via a pandas. 

## Requirements
faiss
transformers
pandas
numpy

## Get Started
Method 1. 
Use the command line (Requires loading data and models every time):
!python main.py -h

Method 2. 
Load each component, see tests/test.py

## Parameters:

index_path (Required) : A string that represents the path to a faiss index containing the items for retrieval. This argument is required and is used to specify the location of the index.

retriever (Required) : A string that specifies the Transformers automodel to be used for embedding for retrieval. The default value is "facebook/contriever-msmarco".

query (Optional) : A string that represents the query for retrieval. The default value is "Who was the last man on the moon?".

device (Optional): A string that specifies the device to run processing on, either "gpu" or "cpu". The default value is "cpu". 

dataset_path (Optional): A string that represents the path to the parquet dataset containing the items for retrieval as strings. The default value is None.

column (Required when using a dataset): A string that specifies the name of the column in the dataset containing the sentences/paragraphs for retrieval. The default value is None. 

extra_columns (Optional): A string that specifies the columns to return alongside the result indices when using a dataset. The default value is None. 

n (Optional): An integer that specifies the number of results to retrieve. The default value is 5.

use_nn (Optional): A Boolean value that determines whether to use 3 nearest neighbours. The default value is False. This argument is optional and is used to specify whether to use nearest neighbours for retrieval.

nn_threshold (Optional): A float value that represents the minimum similarity for 3 nearest neighbours when using use_nn. The default value is 0.7. This argument is optional and is used to specify the minimum similarity for nearest neighbours.

json_path: A string that represents the path to a json file to output the results. The default value is an empty string. This argument is optional and is used to specify the location to output the results to a json file.
