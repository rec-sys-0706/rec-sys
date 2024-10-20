from config import DB, CustomDateTime
from sqlalchemy.orm import mapped_column
from sqlalchemy import Uuid, DateTime, ForeignKey, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class Stopview(DB.Model):
    __tablename__ = 'stop_view'

    uuid = mapped_column(Uuid(as_uuid=True), primary_key=True)
    user_id = mapped_column(Uuid(as_uuid=True), ForeignKey('app_user.uuid'), nullable=False)
    item_id = mapped_column(Uuid(as_uuid=True), ForeignKey('item.uuid'), nullable=False)
    stop_time = mapped_column(DateTime, nullable=False)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class StopviewSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Stopview
        load_instance = False
    stop_time = CustomDateTime()