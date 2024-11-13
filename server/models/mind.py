from config import DB, CustomDateTime
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy import ForeignKey, Uuid, String, Text, Enum, DateTime, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema


class Mind(DB.Model):
    __tablename__ = 'mind'

    item_uuid = mapped_column(Uuid(as_uuid=True), ForeignKey('item.uuid'), primary_key=True)  # 設定 item_uuid 為主鍵
    mind_id = mapped_column(String(50), nullable=False, unique=True)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class MindSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Mind
        load_instance = False