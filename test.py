from BrickPilot import BrickPilot

token = f'{os.environ.get("DATABRICKS_TOKEN")}'
codegen_url = '<code generation model endpoint>'
explain_url = '<code summarization model endpoint>'
#Example prompts here. Can be anything
autocoder.generate_code("Function to Print a message with a given name")
autocoder.explain_code("def hello(name):\n print(name)")