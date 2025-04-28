# api/recommendation_api.py
from fastapi import APIRouter
from api.schemas import RecommendRequest, RecommendResponse, CarRecommendation
from api.services import recommend_cars_for_user

router = APIRouter()

@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    recommendations = recommend_cars_for_user(request.user_id)
    return RecommendResponse(recommendations=recommendations)
