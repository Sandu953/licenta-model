from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    user_id: int

class CarRecommendation(BaseModel):
    maker: str
    genmodel: str
    item_id: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: List[CarRecommendation]
