from config import DB
from sqlalchemy.orm import mapped_column
from sqlalchemy import Uuid, Float, ForeignKey, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class Recommendationscore(DB.Model):
    __tablename__ = 'recommendation_score'

    user_id = mapped_column(Uuid(as_uuid=True), ForeignKey('app_user.uuid'), primary_key=True)
    item_id = mapped_column(Uuid(as_uuid=True), ForeignKey('item.uuid'), primary_key=True)
    score = mapped_column(Float, nullable=False)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class RecommendationscoreSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Recommendationscore
        load_instance = False