from langchain.text_splitter import RecursiveCharacterTextSplitter

def splitter_text():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    return text_splitter