from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from config import CustomDateTime
from server.models.behavior import Behavior


class BehaviorSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Behavior
        load_instance = False
    clicked_time = CustomDateTime()