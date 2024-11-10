import ibm_boto3
from ibm_botocore.client import Config
import os
from dotenv import load_dotenv

load_dotenv()

# IBM Cloud Object Storage credentials
cos_credentials = {
    "apikey": os.getenv("COS_APIKEY"),
    "endpoint": os.getenv("COS_ENDPOINT"),
    "service_instance_id": os.getenv("COS_SERVICE_INSTANCE_ID")
}

# Create a client connection
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=cos_credentials["apikey"],
    ibm_service_instance_id=cos_credentials["service_instance_id"],
    config=Config(signature_version="oauth"),
    endpoint_url=cos_credentials["endpoint"]
)

def retrieve_file_from_cos(bucket_name=os.getenv("COS_BUCKET_NAME"), file_name=""):
    try:
        # Create a file object
        file_object = cos.Object(bucket_name, file_name)
        
        # Retrieve the file
        file_content = file_object.get()['Body'].read()
        
        return file_content
    except Exception as e:
        print(f"An error occurred while retrieving the file: {str(e)}")
        return None
    
def list_files_in_bucket(bucket_name=os.getenv("COS_BUCKET_NAME")):
    try:
        # Get the bucket
        bucket = cos.Bucket(bucket_name)
        
        # List objects in the bucket
        files = []
        for obj in bucket.objects.all():
            files.append(obj.key)

        return files
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        return None

"""
file_name = "Adding an Estimate to an Opportunity.md"
# Specify the bucket name and local file details
bucket_name = os.getenv("COS_BUCKET_NAME")
#print(bucket_name)
#file_content = retrieve_file_from_cos(bucket_name, file_name)
file_list = list_files_in_bucket()
print(file_list)
"""