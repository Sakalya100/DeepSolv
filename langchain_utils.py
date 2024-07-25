import os
import dotenv
from langchain_community.callbacks import get_openai_callback
from pinecone_utils import *
import time

dotenv.load_dotenv(override=True)
db = os.environ["VECTOR_DB"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")

def retrieve(query):
    if db == "Pinecone":
        print("Retrieving context from Pinecone")
        context = retrieve_context_pinecone(PINECONE_INDEX_NAME, query)

    return context

def get_langchain_response(query, session_id):
    if db == "Pinecone":
        print("Starting LangChain retrieval")
        start = time.time()
        print(f"Creating History Aware Rag Chain on index {PINECONE_INDEX_NAME}")
        rag_chain = create_history_aware_rag_chain(PINECONE_INDEX_NAME, session_id, query)
        print("Rag Chain created")
        with get_openai_callback() as cb:
            print("Invoking Rag Chain")
            try:
                result = rag_chain.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": session_id}},
                )
                if isinstance(result, dict):
                    ans = result.get("answer", "No answer found")
                    refined_query = result.get("refined_query", query)
                    retrieved_context = result.get("context", "No context retrieved")
                else:
                    print(f"Unexpected result type: {type(result)}")
                    ans = str(result)
                    refined_query = query
                    retrieved_context = "Unable to retrieve context"
            except Exception as e:
                print(f"Error during rag_chain.invoke: {e}")
                ans = f"An error occurred: {str(e)}"
                refined_query = query
                retrieved_context = "Error occurred during retrieval"
        
        end = time.time()
        total_time = end - start
        input_tokens, output_tokens, total_tokens = cb.prompt_tokens, cb.completion_tokens, cb.total_tokens
        return ans, total_time, input_tokens, output_tokens, total_tokens, refined_query, retrieved_context