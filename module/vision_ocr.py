import re
import fitz
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
from google.cloud import vision
from google.oauth2 import service_account

def vision_ocr_credentials(config):
    my_credentials = config["CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_file(my_credentials)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client

def vision_ai_ocr(file, vision_client):
    config = dotenv_values(".env")
    docs = fitz.open(file)
    json_data = pd.read_json(config["COMPANY_INFO"])
    stock_code = Path(config["FILES"]).stem.split("_")[1]
    company_name = json_data[json_data["stockCode"] == int(stock_code)]
    sentences = []
    for page in range(2,6):
        page = docs.load_page(page)
        mat = fitz.Matrix(2.0, 2.0)
        image = page.get_pixmap(matrix = mat)
        image_bytes = vision.Image(content = image.tobytes())
        response = vision_client.text_detection(image = image_bytes)
        texts = response.text_annotations
        for text in texts:
            text = re.sub(r"[:,.，、]", "", text.description)
            text = text.replace("本公司", company_name.iloc[0]["abbreviation"]).replace("公司", company_name.iloc[0]["abbreviation"])
            sentences.append(text)
    sentences = "".join(sentences)
    return sentences
