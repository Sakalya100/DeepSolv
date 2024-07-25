from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory,RunnablePassthrough
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from  langchain.chains.llm import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from common_utils import format_context, create_chat_history_prompt
from db_utils import get_past_conversation

dotenv.load_dotenv(override=True)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
api_key = os.getenv("OPENAI_API_KEY_2")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")

def initialize_pinecone(index_name):
    print("Initializing Pinecone Vector Store")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore

def create_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def create_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)


def create_rag_chain(retriever, llm, prompt):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
def get_session_history(session_id):
    # return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
    messages = get_past_conversation(session_id)
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "human":
            history.add_user_message(message["content"])
        elif message["role"] == "ai":
            history.add_ai_message(message["content"])
    return history

def retrieve_context_pinecone(pinecone_index_name, query):
    print(f"Retrieving context for query: {query}")
    
    vectorstore = initialize_pinecone(pinecone_index_name)
    print(f"Vectorstore initialized: {vectorstore}")
    
    retriever = create_retriever(vectorstore)
    print(f"Retriever created: {retriever}")
    
    try:
        relevant_docs = retriever.get_relevant_documents(query)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None
    
    print(f"Retrieved {len(relevant_docs)} documents")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i+1}: {doc.page_content[:100]}...")
    
    return format_context(relevant_docs) if relevant_docs else None

def create_history_aware_rag_chain_test(pinecone_index_name, session_id, query):
    vectorstore = initialize_pinecone(pinecone_index_name)
    retriever = create_retriever(vectorstore)
    llm = create_llm()
    contextualize_q_prompt, qa_prompt = create_chat_history_prompt()
    history_aware_retriever = create_history_aware_retriever(
                                llm, retriever, contextualize_q_prompt
                            )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
                                rag_chain,
                                get_session_history, 
                                input_messages_key="input",
                                history_messages_key="chat_history",
                                output_messages_key="answer",
                            )
    
    return conversational_rag_chain


def create_history_aware_rag_chain(pinecone_index_name, session_id, query):
    vectorstore = initialize_pinecone(pinecone_index_name)
    retriever = create_retriever(vectorstore)
    llm = create_llm()
    contextualize_q_prompt, qa_prompt = create_chat_history_prompt()

    # Step 1: Get session history
    chat_history = get_session_history(session_id)
    
    # Step 2: Creating question-answering chain
    question_answer_chain = LLMChain(llm=llm, prompt=qa_prompt)

    # Step 3: Create refined query
    def refine_query(inputs):
        print("Getting refined query")
        chat_messages = [
            HumanMessage(content=msg) if isinstance(msg, str) else 
            HumanMessage(content=msg.content) if isinstance(msg, HumanMessage) else
            AIMessage(content=msg.content)
            for msg in inputs["chat_history"].messages[-5:]  # Use last 5 messages
        ]
        refined = llm.invoke(contextualize_q_prompt.format(chat_history=chat_messages, input=inputs["input"]))
        print(f"Refined query: {refined.content}")
        return refined.content if isinstance(refined, AIMessage) else refined

    # Step 4: Use history-aware retriever
    def retrieve_docs(refined_query):
        print("Retrieving relevant documents")
        retrieved_docs = retriever.invoke(refined_query)
        print(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    # Step 5: Question answering
    def answer_question(inputs):
        print("Answering question")
        print(inputs)
        docs = inputs["docs"]
        query = inputs["query"]  
        chat_history = inputs["chat_history"].messages
        response = question_answer_chain.invoke({"context": docs, "input": query, "chat_history":chat_history})
        print(f"Answer: {response['text']}")
        return response['text']  

    # Setp 6: Final Chain creation
    rag_chain = (
        RunnablePassthrough.assign(chat_history=lambda _: chat_history) # Assigning chat_history fetched in Step 1
        | RunnablePassthrough.assign(refined_query=refine_query) # Creating refined query with chat_history and query
        | RunnablePassthrough.assign(docs=lambda x: retrieve_docs(x["refined_query"])) # Retrieving relevant docs using refined query
        | RunnablePassthrough.assign(
            answer=lambda x: answer_question({"docs": x["docs"], "query": x["input"], "chat_history":chat_history}) # Generating answer based on docs, chat_history and user_query
        )
        | (lambda x: {
            "answer": x["answer"],
            "refined_query": x["refined_query"],
            "context": x["docs"]
        })
    )

    return rag_chain

def find_and_store_chunk_ids(index_name: str, pdf_name: str):
    """
    Search for chunks in a Pinecone index with a specific PDF name in the "source" metadata
    and return their IDs.

    :param index_name: Name of the Pinecone index
    :param pdf_name: Name of the PDF file to search for in the "source" metadata
    :return: List of chunk IDs
    """
    try:
        # Initialize Pinecone (make sure you've set up your API key)
        pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        )

        # Connect to the Pinecone index
        index = pc.Index(index_name)

        index_stats = index.describe_index_stats()
        print(f"Index stats: {index_stats}")

        query_filter = {"source": {"$eq": f"temp_files\\{pdf_name}"}}

        results = index.query(vector=[0] * index_stats['dimension'], filter=query_filter,top_k=1000)

        chunk_ids = [match.id for match in results.matches]

        print(f"Query filter: {query_filter}")
        print(f"Found {len(chunk_ids)} chunks")

        print(f"Total unique chunks found: {len(chunk_ids)}")
        return chunk_ids

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []
    
def delete_doc_from_pinecone(filename):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    chunk_ids = find_and_store_chunk_ids(index_name, filename)
    print(chunk_ids)
    try:
        print(f"Deleting {filename} from index {index_name}")
        index.delete(ids=chunk_ids)
        return True
    except Exception as e:
        print(f"Error deleting {filename} from index {index_name}: {str(e)}")
        return False
    