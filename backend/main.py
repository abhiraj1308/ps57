from fastapi import FastAPI
app = FastAPI(title="PS57 API")

@app.get("/health")
def health():
    return {"status": "ok"}