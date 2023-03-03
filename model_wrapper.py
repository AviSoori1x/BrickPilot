from transformers import AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
from sys import version_info

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py")


tokenizer_path = '/tmp/tokenizer/'
model_path = '/tmp/model/'
tokenizer.save_pretrained(tokenizer_path)
model.save_pretrained(model_path)

artifacts = {
 
  "model":model_path,
  "tokenizer": tokenizer_path
}

#Tests to see the text input is valid 

payload_pd = pd.DataFrame([["# Function to print hello world"]],columns=['text'])

import mlflow.pyfunc

class CodeGenerator(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['tokenizer'])
    self.model = T5ForConditionalGeneration.from_pretrained(context.artifacts['model'])
  def predict(self, context, model_input ):
    import json
    texts = model_input.iloc[:,0].to_list() # get the first column
    input_ids = self.tokenizer(texts, return_tensors="pt").input_ids
    # simply generate a single sequence
    generated_ids = self.model.generate(input_ids, max_length=512)
    generated_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    result = {'code': generated_code}
    return json.dumps(result)
  
 
PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

input_example = payload_pd

import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [vscode
          'mlflow',
          'transformers',
          'pandas',
          'cloudpickle=={}'.format(cloudpickle.__version__),
          'torch'],
      },
    ],
    'name': 'code_env'
}

mlflow_pyfunc_model_path = "code_gen_0"

mlflow.set_experiment("/Users/<username>/<subdirectory>/brickbots_gen")

with mlflow.start_run(run_name="brickbots_gen_run") as run:
    mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=CodeGenerator(),artifacts=artifacts,
            conda_env=conda_env, input_example = input_example)
    

class CodeExplainer(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    from transformers import RobertaTokenizer, T5ForConditionalGeneration
    self.tokenizer = RobertaTokenizer.from_pretrained(context.artifacts['tokenizer'])
    self.model = T5ForConditionalGeneration.from_pretrained(context.artifacts['model'])
  def predict(self, context, model_input ):
    import json
    texts = model_input.iloc[:,0].to_list() # get the first column
    input_ids = self.tokenizer(texts, return_tensors="pt").input_ids
    # simply generate a single sequence
    generated_ids = self.model.generate(input_ids, max_length=64)
    generated_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    result = {'explanation': generated_code}
    return json.dumps(result)
  
#Tests to see the text input is valid 

payload_pd = pd.DataFrame([["""for i in range(10):
            print(i)
        """]],columns=['code'])

mlflow_pyfunc_model_path = "code_explain_0"

mlflow.set_experiment("/Users/<username>/<subdirectory>/brickbots_exp")

with mlflow.start_run(run_name="brickbots_exp_run") as run:
    mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=CodeExplainer(),artifacts=artifacts,
            conda_env=conda_env, input_example = input_example)
