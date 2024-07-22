import os
from .splitter_sentence import splitter_text
from model.embedding_model import embedding_model
from module.vision_ocr import vision_ai_ocr, vision_ocr_credentials
from langchain_community.vectorstores import Chroma

def vector_store_document(document, config):
    embedding = embedding_model()
    persist_directory = config["SAVE_DB_PATH"]
    table_name = config["SAVE_TABLE_NAME"]
    if os.path.isdir(persist_directory):
        print("Get Collection")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
            collection_name=table_name
        )
        vectordb.get()
    else:
        print("Create Collection")
        # Create a Chroma vectorstore using LangChain, this represents the collection
        vectordb = Chroma.from_documents(
            documents=document, 
            embedding=embedding, 
            persist_directory=persist_directory,
            collection_name=table_name
        )
    return vectordb

def vector_store_text(file, config):
    embedding = embedding_model()
    persist_directory = config["SAVE_DB_PATH"]
    table_name = config["SAVE_TABLE_NAME"]
    if os.path.isdir(persist_directory):
        print("Get Collection")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
            collection_name=table_name
        )
        vectordb.get()
    else:
        print("Create Collection")
        vision_client = vision_ocr_credentials(config)
        sentences = vision_ai_ocr(file, vision_client)
        text_splitter = splitter_text()
        all_splits = text_splitter.split_text(sentences)
        vectordb = Chroma.from_texts(
            texts=all_splits, 
            embedding=embedding, 
            persist_directory=persist_directory,
            collection_name=table_name
        )
    return vectordb