import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec

index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Initialize HuggingFace SentenceTransformer embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Create the index if it doesn't exist
print("Initializing or get existing Pinecone index")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)
print("Index initialized")

# Function to load and process documents
def load_and_process(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Load PDF
print("Loading PDF")
pdf_url = "https://www.apple.com/privacy/docs/Apple_Vision_Pro_Privacy_Overview.pdf"
pdf_docs = load_and_process(PyPDFLoader(pdf_url))
print("PDF Loaded")

# Load website
print("Loading Website")
web_url = "https://www.apple.com/apple-vision-pro/"
web_docs = load_and_process(WebBaseLoader(web_url))
print("Website Loaded")

# Load YouTube video
print("Youtube Video")
youtube_url = "https://www.youtube.com/watch?v=TX9qSaGXFyg"
youtube_docs = load_and_process(YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True))
print("Youtube Video Loaded")

# Combine all documents
print("combining all docs")
all_docs = pdf_docs + web_docs + youtube_docs

# Create and populate Pinecone index
print("Creating vectorstore from docs")
vectorstore = PineconeVectorStore.from_documents(all_docs, embeddings, index_name=index_name)

print("Data successfully loaded into Pinecone index.")