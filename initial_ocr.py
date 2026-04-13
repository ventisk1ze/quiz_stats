import os
import json
import time
import base64
from functools import wraps
from httpx import ReadTimeout, ConnectTimeout
from mistralai.client import Mistral
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

def retry(max_retries=3, delay=1, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_retries=1000, delay=2, exceptions=(ReadTimeout, ConnectTimeout))
def process():
    # Requests sometime result in error so we take a snapshot of responses folder
    processed = [Path(p).stem for p in os.listdir('responses')]
    print(f'Responses from the last try: {len(processed)}')

    docs = [
        [{
            "type": "image_url",
            "image_url": f'data:image/jpeg;base64,{encode_image_to_base64(f"data/{f}")}'
        },
        f] for f in os.listdir('data') if Path(f).stem not in processed
    ]

    print(f'Docs in this try: {len(docs)}')


    for doc in docs:
        resp = client.ocr.process(
            model="mistral-ocr-latest",
            document=doc[0],
            extract_header=True,
            include_image_base64=True,
            timeout_ms= 20 * 1000 # milliseconds
        )

        with open(f'responses/{Path(doc[-1]).stem}.json', 'w+', encoding='utf8') as f:
            json.dump(resp.model_dump_json(), f, indent=4, ensure_ascii=False)

        time.sleep(5)

if __name__ == '__main__':
    process()
    print(f'Processed {len(os.listdir("responses"))} files')