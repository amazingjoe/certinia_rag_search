# Connect to Milvus
import sys, os
import dotenv
# Load environment variables from a .env file
dotenv.load_dotenv()
from pymilvus import connections,utility,MilvusClient, DataType, CollectionSchema, FieldSchema, DataType, Collection
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.core import Settings

def get_milvus_client():
    print("start connecting to Milvus")
    connections.connect(
    host="e6fa5694-3b5f-47ec-a98a-2f749d5a1aab.cqh2jh8d00ae3kp0jmpg.lakehouse.appdomain.cloud", 
    port="31398", 
    secure=True, 
    user="ibmlhapikey", 
    password="xxLKzTzutt69KkeNHCs0Keihq00niCCFHLzzMky8xrVp"
    )
    return utility


def search_milvus(milvus_client,collection_name,query_vector):
    collection = get_milvus_collection(milvus_client,collection_name)
    search_params = {
        "metric_type": "COSINE",
        "params": {},
    }    
    return collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=1,
        output_fields=["id", "document_name", "bucket", "document_chunk"]
    )

def query_milvus(milvus_client,collection_name):
    collection = get_milvus_collection(milvus_client,collection_name)
    return collection.query(
        expr="",
        limit=5,
        output_fields=["id", "document_name", "bucket"]
    ) 

def create_milvus_collection(milvus_client,collection_name):
    # Define the primary key field
    primary_field = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True  # Milvus will auto-generate unique IDs
    )

    # Define the vector field
    vector_field = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=768  # Dimension matching the model's output
    )

    # Define the document_name field
    document_name_field = FieldSchema(
        name="document_name",
        dtype=DataType.VARCHAR,
        max_length=255  # Adjust the length as needed
    )

    # Define the bucket field
    bucket_field = FieldSchema(
        name="bucket",
        dtype=DataType.VARCHAR,
        max_length=512  # Adjust the length as needed
    )

    # Define the document_chunk field
    document_chunk_field = FieldSchema(
        name="document_chunk",
        dtype=DataType.VARCHAR,
        max_length=2048  # Adjust the length as needed
    )    

    # Create the collection schema
    schema = CollectionSchema(
        fields=[primary_field, vector_field, document_chunk_field, document_name_field, bucket_field],
        description="Certinia Estimates RAG"
    )

    # Create the collection
    Collection(name=collection_name, schema=schema)
    print(f"Collection {collection_name} created successfully")

def get_milvus_collection(milvus_client,collection_name):
    #milvus_client = get_milvus_client()
    return Collection(collection_name)

def get_milvus_collection_schema(milvus_collection):
    return milvus_collection.schema

def insert_into_milvus(milvus_collection, data):
    try:    
        milvus_collection.insert(data)
    except Exception as e:
        print(f"Error inserting data into Milvus: {e}")

def create_milvus_index(milvus_collection, index_name, index_params):
    #index_params = {
    #    "index_type": "IVF_FLAT",
    #    "metric_type": "COSINE",
    #    "params": {"nlist": 128},
    #}    
    milvus_collection.create_index(index_name, index_params)
    print(f"Index {index_name} created successfully")

def delete_milvus_index(collection_name, index_name):
    milvus_collection = get_milvus_collection(collection_name)
    milvus_collection.drop_index(index_name)

def get_embedding(chunk_text):
    embed_model = WatsonxEmbeddings(model_id=os.getenv("EMBED_MODEL_ID"),
        url=os.getenv("EMBED_URL"),
        apikey=os.getenv("EMBED_APIKEY"),
        project_id=os.getenv("EMBED_PROJECT_ID"))

    embedding = embed_model.get_text_embedding(chunk_text)
    # Ensure the embedding is of the correct dimension
    if len(embedding) != 768:
        raise ValueError(f"Embedding dimension mismatch: expected 768, got {len(embedding)}")
    
    return embedding

def dump_milvus_collection(milvus_collection):
    # Define the number of embeddings to retrieve
    milvus_collection.load()

    # Example search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }    

    query_vector = get_embedding("How do I scope a new project?")   

    # Perform the search
    results = milvus_collection.search(
        data=query_vector,
        anns_field="embedding",  # Specify the field to search
        param=search_params,
        limit=10
    )
    
    # Process the results as needed
    for result in results:
        print(result)


