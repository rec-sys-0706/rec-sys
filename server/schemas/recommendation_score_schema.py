from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from server.models.recommendation_score import Recommendationscore


class RecommendationscoreSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Recommendationscore
        load_instance = False