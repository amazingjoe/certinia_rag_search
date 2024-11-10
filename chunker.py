"""
chunker.py - A script to create and persist a vector index from documents.

Usage:
    python chunker.py

"""

import os
import dotenv
# Load environment variables from a .env file
dotenv.load_dotenv()

import shutil
from ibmcos_utilitites import list_files_in_bucket, retrieve_file_from_cos

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext,load_index_from_storage
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.core import Settings
from MilvusUtilities import insert_into_milvus, get_milvus_client, get_milvus_collection

def chunk_from_object_store(bucketname):
    embed_model = WatsonxEmbeddings(model_id=os.getenv("EMBED_MODEL_ID"),
                        url=os.getenv("EMBED_URL"),
                        apikey=os.getenv("EMBED_APIKEY"),
                        project_id=os.getenv("EMBED_PROJECT_ID"))

    # Set global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = 350  # Adjust this based on your needs and model's limitations

    # Create the 'docs' directory if it does not exist
    if not os.path.exists("docs"):
        os.makedirs("docs")

    # List all files in the specified bucket
    file_list = list_files_in_bucket()

    bucketname = os.getenv("COS_BUCKET_NAME")

    # Download each file into the 'docs' directory
    for file_name in file_list:
        file_content = retrieve_file_from_cos(bucket_name=bucketname, file_name=file_name)
        if file_content:
            with open(os.path.join("docs", file_name), "wb") as file:
                file.write(file_content)

    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)
    nodes = index.docstore.docs.values()

    counter = 0
    datarows = []
    for node in nodes:
        counter += 1
        chunk_text = node.text
        file_path = node.metadata['file_path']
        file_name = node.metadata['file_name']
        embedding = embed_model.get_text_embedding(chunk_text)
        datarow = {"embedding": embedding, "document_name": file_name, "bucket": bucketname, "document_chunk": chunk_text}
        datarows.append(datarow)

    milvus_client = get_milvus_client()
    milvus_collection = get_milvus_collection(milvus_client, "CertiniaEstimatesRAG")
    insert_into_milvus(milvus_collection, datarows)
