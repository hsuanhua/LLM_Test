from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def embedding_model():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    # embedding= GoogleGenerativeAIEmbeddings(
    #     model="models/text-embedding-004",
    #     google_api_key=os.getenv("GOOGLE_API_KEY")
    # )
    return embedding