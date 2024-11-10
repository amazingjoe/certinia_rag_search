#from openai import OpenAI
from llama_index.llms.ibm import WatsonxLLM
from llama_index.core import Settings
import os
from dotenv import load_dotenv

load_dotenv()


def query_llm(context,question):
    print("Running query_llm") 

    template = f"<|system|>You are a helper bot that takes a question and a chunk of information text and uses that to provide an answer for the user<|system|>\n<|question|>{question}<|question|>\n<|information|>{context}<|information|>\n"

    # Set up the IBM WatsonX API key using environment variables
    Settings.llm = WatsonxLLM(model_id=os.getenv("MODEL_ID"),
                        url=os.getenv("URL"),
                        apikey=os.getenv("APIKEY"),
                        project_id=os.getenv("PROJECT_ID"))    
  
    # Send a request to the LLM
    response = Settings.llm.complete(
        prompt=f"{template}\n",
        max_new_tokens=4028,
        temperature=1.0
    )    
    
    #return completion.choices[0].message.content
    return response.text.strip()