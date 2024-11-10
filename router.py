"""
This Flask application provides an endpoint to query a language model (LLM).

Usage:
1. Start the Flask server by running this script.
2. Send a POST request to the '/query' endpoint with a JSON payload containing a 'question' key.
3. The server will attempt to classify the question by querying the LLM up to 3 times.
4. If the LLM returns a classification other than "Retry", the classification is returned in the response.
5. If no valid classification is obtained after 3 attempts, the last classification is returned.

Example request:
POST /query
{
    "question": "What is the capital of France?"
}

Error Handling:
- If the 'question' key is missing in the request payload, a 400 error is returned with an error message.
"""
from waitress import serve
from flask import Flask, request, jsonify
from llmcall import query_llm, clean_response
from rag_search_estimates import rag_search, rag_search_and_answer
import json

app = Flask(__name__)

from chunker import chunk_from_object_store
import os
import dotenv

dotenv.load_dotenv()

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy"}), 200


@app.route('/chunk_object_files', methods=['GET'])
def chunk_object_files():
    bucketname = os.getenv("COS_BUCKET_NAME")

    if not bucketname:
        return jsonify({"error": "Bucket name is not configured"}), 500

    try:
        chunk_from_object_store(bucketname)
        return jsonify({"message": "Files chunked and indexed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    counter = 0

    if not question:
        return jsonify({"error": "Question is required"}), 400

    while counter < 3:
        response = query_llm(question, counter)
        classification = clean_response(response, counter)
        if classification != "Retry":
            break
        counter += 1

    rag_response = rag_search_and_answer(question)
    rag_response_obj = json.loads(rag_response)
    rag_response_obj["classification"] = classification
    #return jsonify({"classification": classification})
    return rag_response_obj

if __name__ == '__main__':
    print("Starting server...")
    serve(app, host='0.0.0.0', port=8080)
