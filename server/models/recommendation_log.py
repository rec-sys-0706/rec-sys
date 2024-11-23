from config import DB, CustomDateTime
from sqlalchemy.orm import mapped_column
from sqlalchemy import Float, Uuid, DateTime, Boolean, ForeignKey, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class Recommendationlog(DB.Model):
    __tablename__ = 'recommendation_log'

    uuid = mapped_column(Uuid(as_uuid=True), primary_key=True)
    user_id = mapped_column(Uuid(as_uuid=True), ForeignKey('app_user.uuid'), nullable=False)
    item_id = mapped_column(Uuid(as_uuid=True), ForeignKey('item.uuid'), nullable=False)
    recommend_score = mapped_column(Float(precision=4), nullable=False)
    recommend_datetime = mapped_column(DateTime, nullable=False)
    is_recommend = mapped_column(Boolean, default=False, nullable=False)
    # gattered_datetime = mapped_column(DateTime, nullable=False)
    # clicked = mapped_column(Boolean, default=False)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class RecommendationlogSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Recommendationlog
        load_instance = False
    gattered_datetime = CustomDateTime()
