from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from config import CustomDateTime
from server.models.recommendation_log import Recommendationlog


class RecommendationlogSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Recommendationlog
        load_instance = False
    recommend_time = CustomDateTime()