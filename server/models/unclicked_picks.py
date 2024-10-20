from config import DB, CustomDateTime
from sqlalchemy.orm import mapped_column
from sqlalchemy import Uuid, DateTime, ForeignKey, Boolean, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class Unclickedpicks(DB.Model):
    __tablename__ = 'unclicked_picks'

    uuid = mapped_column(Uuid(as_uuid=True), primary_key=True)
    user_id = mapped_column(Uuid(as_uuid=True), ForeignKey('app_user.uuid'), nullable=False)
    item_id = mapped_column(Uuid(as_uuid=True), ForeignKey('item.uuid'), nullable=False)
    recommend_time = mapped_column(DateTime, nullable=False)
    status = mapped_column(Boolean, default=False, nullable=False)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class UnclickedpicksSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Unclickedpicks
        load_instance = False
    recommend_time = CustomDateTime()
