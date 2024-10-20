from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from config import CustomDateTime
from server.models.item import Item


class ItemSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Item
        load_instance = False # load as SQLAlchemy instance or plain dict.
    gattered_datetime = CustomDateTime()