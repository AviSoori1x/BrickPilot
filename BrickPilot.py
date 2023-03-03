import os
import requests
import numpy as np
import pandas as pd
import json

class BrickPilot():
  
  def __init__(self, token, codegen_url, explain_url):
    self.token = token
    self.codegen_url = codegen_url
    self.explain_url = explain_url
    
  def create_tf_serving_json(self, data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

  def score_model(self, dataset, task):
    if task=='generate':
      url = self.codegen_url
    if task=='explain':
      url = self.explain_url

    headers = {'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()
  
   #Compose the code generation function 
  def generate_code(self, input):
    payload_pd = pd.DataFrame([["# " +input]],columns=['text'])
    result = self.score_model(payload_pd, 'generate')
    return print(json.loads(result['predictions'])['code'])

   #Compose the explanation function 
  def explain_code(self, input):
    payload_pd = pd.DataFrame([["# " +input]],columns=['text'])
    result = self.score_model(payload_pd, 'explain')
    return print(json.loads(result['predictions'])['explanation'])