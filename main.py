# main.py
from fastapi import FastAPI
from api.recommendation_api import router as recommendation_router

app = FastAPI(
    title="Car Recommendation API",
    description="Recommends cars based on recent interactions",
    version="1.0.0"
)

app.include_router(recommendation_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


