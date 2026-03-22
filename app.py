"""
app.py — Servidor web para o demonstrativo interativo.

Roda com:
    python app.py
    # Abre http://localhost:8000 automaticamente
"""

import webbrowser
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
