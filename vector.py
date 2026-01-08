from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df=pd.read_csv("realistic_restaurant_reviews.csv")
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

db_location="./chroma_langchain_db"
add_documents=not os.path.exists(db_location)   

if add_documents:
    documents = []
    ids = []
    # Convert each row in the CSV into a Document with metadata
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    
# Create or connect to the Chroma vector database
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
# Add documents to the vector store if it’s a new DB
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
# Create a retriever for searching similar documents
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Return top 5 similar documents
)