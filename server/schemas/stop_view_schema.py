from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from config import CustomDateTime
from server.models.stop_view import Stopview


class StopviewSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Stopview
        load_instance = False
    stop_time = CustomDateTime()