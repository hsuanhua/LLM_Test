from langchain_google_genai import GoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

def llm_chain(config):
    GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
    sql_model = config["CHAIN_MODEL"]
    llm = GoogleGenerativeAI(
        model=sql_model, temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    )
    
    return llm
