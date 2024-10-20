from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from config import CustomDateTime
from server.models.unclicked_picks import Unclickedpicks


class UnclickedpicksSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Unclickedpicks
        load_instance = False
    recommend_time = CustomDateTime()