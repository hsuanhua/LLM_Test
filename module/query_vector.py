from .create_vectordb import vector_store_document, vector_store_text
from model.llm_model import llm_chain
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate


def split_text_embedding(file, config):
    vectordb = vector_store_text(file, config)
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k":4
        }
    )

    template = """You are an intelligent assistant responsible for organizing company information.
    From the query, you analyze and extract the information from database.
    And return detailed response by trandition Chinese.

    {context}

    Question: {question}
    AI:"""
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm_chain(config), 
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs = {"prompt": PROMPT}
    )
    query = config["QUERY"]
    response = retrievalQA.invoke(query)
    return response

def split_json_embedding(json_documents, config):
    vectordb = vector_store_document(json_documents, config)
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":int(len(json_documents)*0.4)
        }
    )
    template = """You are an intelligent assistant responsible for organizing company information.
    From the query, analyze and extract the company name, and then search the database for relevant company information.
    And return response by trandition Chinese.

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm_chain(config), 
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs = {"prompt": PROMPT}
    )
    query = config["QUERY"]
    response = retrievalQA.invoke(query)
    return response