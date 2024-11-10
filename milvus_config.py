"""
This script is meant to be run once when a new collection is needed, it establishes the collection and then creates it with this schema.

Schema:
- id: INT64, primary key, auto-generated
- embedding: FLOAT_VECTOR, dimension 768
- document_name: VARCHAR, max length 255
- bucketname: VARCHAR, max length 512
- document_chunk: VARCHAR, max length 512
"""

from MilvusUtilities import create_milvus_collection,get_milvus_client, get_milvus_collection, create_milvus_index

milvus_client = get_milvus_client()
if milvus_client.has_collection("CertiniaEstimatesRAG"):
    milvus_client.drop_collection("CertiniaEstimatesRAG")

milvus_collection = create_milvus_collection(milvus_client,"CertiniaEstimatesRAG")

collection = get_milvus_collection(milvus_client,"CertiniaEstimatesRAG")

index_params = {
    "index_type": "IVF_FLAT",  # Choose an appropriate index type
    "params": {"nlist": 128},  # Adjust parameters as needed
    "metric_type": "COSINE"  # Use Cosine similarity
}

create_milvus_index(collection, "embedding",index_params)