from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.prompts import MessagesPlaceholder
from prompts import review_template_str, refine_query_prompt, question_answering_prompt

def create_chat_history_prompt():
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", refine_query_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_answering_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
    
    return contextualize_q_prompt, qa_prompt

def format_context(relevant_docs):
    formatted_context = ""
    for i, doc in enumerate(relevant_docs, 1):
        formatted_context += f"Document {i}:\n"
        formatted_context += f"Content: {doc.page_content}\n"
        if doc.metadata:
            formatted_context += "Metadata:\n"
            for key, value in doc.metadata.items():
                formatted_context += f"  {key}: {value}\n"
        formatted_context += "\n"
    return formatted_context.strip()

def get_page_content(retrieved_context):
    context = ""
    for doc in retrieved_context:
            context += doc.page_content
            context += "\n"
    return context
