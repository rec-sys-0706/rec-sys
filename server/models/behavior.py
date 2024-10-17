from config import DB, CustomDateTime
from sqlalchemy.orm import mapped_column
from sqlalchemy import UUID, DateTime, ForeignKey, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class Behavior(DB.Model):
    # __tablename__ = 'behavior'

    uuid = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id = mapped_column(UUID(as_uuid=True), ForeignKey('app_user.uuid'), nullable=False)
    item_id = mapped_column(UUID(as_uuid=True), ForeignKey('item.uuid'), nullable=False)
    clicked_time = mapped_column(DateTime, nullable=False)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
    
class BehaviorSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Behavior
        load_instance = False
    clicked_time = CustomDateTime()