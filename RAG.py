#Text extraction and embeddings dependencies
from flask import Flask, request, jsonify
import PyPDF2
import io

#Redis dependencies
import numpy as np
import pandas as pd
import redis
import requests
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

#Sentence Transformers
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
import os

import fireworks.client

# url = "https://didulglohplnliryyzns.supabase.co/storage/v1/object/sign/pdf/88e3bac4-13e3-4d29-9962-57fe95758329/2307.06435.pdf?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJwZGYvODhlM2JhYzQtMTNlMy00ZDI5LTk5NjItNTdmZTk1NzU4MzI5LzIzMDcuMDY0MzUucGRmIiwiaWF0IjoxNzA5Mjk1NTYxLCJleHAiOjE3MDk5MDAzNjF9.r1_x3GJbm6k43dG1Y8PeziSjc44X6YO6pWKC4j0MMAY&t=2024-03-01T12%3A19%3A21.520Z"
fireworks.client.api_key = "iaMTlSAgZeJnVAGOzKURAkw5Ux6NweLIrRxdBT80eycdcqXt"

def str_to_doc(name):
   folder_name = 'docs'
   if not os.path.exists(folder_name):
       return "error"
   file_name = name+'.txt'
   path = os.path.join(folder_name, file_name)
#    with open(path, "w") as file:
#         file.write(text)
   loader = TextLoader(path)
   return loader.load()

def extract_text(url):
  '''
  Extracts data from the supabase public url, splits the text into chunks
  and returns the list of chunks

  '''
#   response = requests.get(url)
#   pdf_content = io.BytesIO(response.content)
#   pdf_reader = PyPDF2.PdfReader(pdf_content)
#   text = ''
#   for page_number in range(len(pdf_reader.pages)):
#     text += pdf_reader.pages[page_number].extract_text()

#   text = text.encode('ascii', 'backslashreplace').decode('ascii', 'ignore')
  #print(text)
  docs = str_to_doc("test")
  
  return docs

def askLLM(prompt):
    completion = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    messages=[
      {
        "role": "user",
        "content": prompt,
      }
    ],
    n=1,
    max_tokens=4000,
    temperature=0.1,
    top_p=0.9,
   )
    return completion.choices[0].message.content

client = redis.Redis(host="redis-19988.c251.east-us-mz.azure.cloud.redislabs.com", port=19988, decode_responses=True, password = "OSzz3Wb89vOVsTceZZWyYPMXMVY1stgB")
def cache_to_redis(client,docs,user_id):
  pipeline = client.pipeline()
  for i, description in enumerate(docs, start=1):
      content = {
          "description":description.page_content
      }
      redis_key = f"{user_id}:{i:03}"
      pipeline.json().set(redis_key, "$", content)
  res = pipeline.execute()
  return res

def sort_by_keys(user_id,client):
  keys = sorted(client.keys(f"{user_id}:*"))
  return keys

embedder = SentenceTransformer("msmarco-distilbert-base-v4")
def generate_embeddings_to_redis(client,keys,user_id,embedder):
  descriptions = client.json().mget(keys, "$.description")
  descriptions = [item for sublist in descriptions for item in sublist]
  embeddings = embedder.encode(descriptions).astype(np.float32).tolist()
  pipeline = client.pipeline()
  for key, embedding in zip(keys, embeddings):
      pipeline.json().set(key, "$.description_embeddings", embedding)
  res = pipeline.execute()
  VECTOR_DIMENSION = len(embeddings[0])
  schema = (
    TextField("$.description", as_name="description"),
    VectorField(
        "$.description_embeddings",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIMENSION,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
  definition = IndexDefinition(prefix=[f"{user_id}:"], index_type=IndexType.JSON)
  res = client.ft(user_id).create_index(
      fields=schema, definition=definition
  )
  return res

def get_info(client,user_id):
  info = client.ft(f"{user_id}").info()
  num_docs = info["num_docs"]
  indexing_failures = info["hash_indexing_failures"]
  print(num_docs,indexing_failures)


def encode_query(user_query, embedder):
   encoded_queries = embedder.encode(user_query).astype(np.float32).tolist()
   return encoded_queries

app = Flask(__name__)



@app.route('/embed', methods=['POST'])
def submit_url():
    data = request.get_json()
    if 'user_id' not in data or 'url' not in data:
        return jsonify({'error': 'user_id and url are required'}), 400
    
    user_id = data['user_id']
    download_url = data['url']
    print(user_id)
    print(download_url)
    try:
        extracted_text = extract_text(download_url)
        cache_to_redis(client,extracted_text,user_id)
        keys = sort_by_keys(user_id,client)
        res = generate_embeddings_to_redis(client,keys,user_id,embedder)
    
        return jsonify({'message': res}), 200
    except:
       return jsonify({'message': "error"}), 500

@app.route("/search",methods = ['POST'] )   
def get_response():
    data = request.get_json()
    if 'user_id' not in data or 'message' not in data:
        return jsonify({'error': 'user_id and url are required'}), 400
    user_id = data['user_id']
    message = data['message']
    encoded_queries = encode_query(message,embedder)
    query = (
    Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'description')
     .dialect(2)
)
    similarity_search = client.ft(user_id).search(query,{"query_vector": np.array(encoded_queries, dtype=np.float32).tobytes()}).docs
    model_context = ""
    for i in similarity_search:
      model_context += i.description
    if user_id not in client.execute_command('FT._LIST'):
       return jsonify({'message': "No vector index found"}), 500
    prompt = f"""You are a smart assistant that answers user questions based on the context provided

user_query: {message}

Answer based on the following context:

data: {model_context}

If the context provided does not answer user query, you can answer outside the context but you have to warn user about it

"""
    completion = askLLM(prompt)
    return jsonify({'completion': completion})

@app.route("/refresh",methods=["GET"])
def refresh():
  try:
    client.flushdb()
    return jsonify({'Status': "Ok"})
  except:
     return jsonify({'error': "error"})

if __name__ == '__main__':
    app.run(host="localhost", debug=True)