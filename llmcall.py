from llama_index.llms.ibm import WatsonxLLM
from llama_index.core import Settings
import os

def query_llm(question, counter):
    print("Running query_llm") 

    template = f"<|system|>Classify responses as one of the following for assisting with a Salesforce application:\n\nCertinia Estimates Helper - The user is asking a question related to how to use the application, procedures for usage of the software or company procedures.\nEstimate Metadata - The user has a question about a field, related object or related object field to the Estimate object.\nOther\n\nPlease respond only with the classification.<|system|>\n<|user|>{question}<|user|>\n"

    # Set up the WatsonxLLM API key
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

def clean_response(response, counter):
    classifications = ["Certinia Estimates Helper", "Estimate Metadata", "Other"]
    
    for classification in classifications:
        if classification in response:
            return classification
    
    counter += 1
    
    if counter > 3:
        return "Error"
    else:
        return "Retry"

