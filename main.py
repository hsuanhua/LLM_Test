import argparse
import pandas as pd
from dotenv import dotenv_values
from langchain_core.documents import Document

from module.query_vector import split_text_embedding, split_json_embedding


def parse():
    parser = argparse.ArgumentParser(description="Process ..")
    parser.add_argument("--txt", action="store_true", help="pdf to ocr")
    parser.add_argument("--pdf", action="store_true", help="pdf to ocr")
    parser.add_argument("--json", action="store_true", help="files")
    opt = parser.parse_args()
    return opt

def main(opt):
    config = dotenv_values(".env")
    file = config["FILES"]
    if opt.pdf:
        response = split_text_embedding(file, config)
    
    if opt.json:
        json_documents = []
        json_data = pd.read_json(file)

        for index, row in json_data.iterrows():
            content = f'{row["abbreviation"]}的資訊: {row["companyName"]}股票代碼為{row["stockCode"]}, 英文簡稱為{row["englishAbbreviation"]} \n'
            metadata = row.drop(labels=["companyName"]).to_dict()
            doc = Document(
                page_content=content,
                metadata=metadata,
            )
            json_documents.append(doc)
        
        response = split_json_embedding(json_documents, config)

    print(response["result"])

if __name__=="__main__":
    opt = parse()
    main(opt)
