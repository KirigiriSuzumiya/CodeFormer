from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import time
from inference_codeformer import inference_entrypoint
import uvicorn
import requests


app = FastAPI()

@app.get("/")
def hello():
    return "hello"


# file: Annotated[bytes, File()],
#     fileb: Annotated[UploadFile, File()],
#     token: Annotated[str, Form()]

@app.get("/upsampler/")
def upsampler_api(url: str):
    imgfile = requests.get(url).content
    file_format = url[url.rfind("."):]
    now = time.time()
    input_path = f"./input_images/{now}"+file_format
    with open(input_path,"wb") as fp:
        fp.write(imgfile)
    output_path = inference_entrypoint(input_path)
    return FileResponse(output_path)

if __name__ == '__main__':
    uvicorn.run(app="cf_app:app", host="0.0.0.0", port=8000, reload=True)
    
    