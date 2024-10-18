from apiflask import APIFlask, Schema
from apiflask.fields import Integer, String, UUID, DateTime
from apiflask.validators import Length, OneOf
from flask_sqlalchemy import SQLAlchemy

class UserIn(Schema):
    account = String(required=True, validate=Length(0, 64))
    password = String(required=True, validate=Length(0, 128))
    email = String(required=True, validate=Length(0, 128))
    line_id = String(required=False, validate=Length(0,64))

class BehaviorIn(Schema):
    user_id = UUID(required=True, as_uuid=True)
    item_id = UUID(required=True, as_uuid=True)
    clicked_time = DateTime(required=True)

class ItemIn(Schema):
    title = String(required=True, validate=Length(500))
    abstract = String(required=True)
    link = String(validate=Length(255))
    data_source = String(validate=(OneOf(['mind_small',
                                     'hf_paper',
                                     'cnn_news'])), required=True)
    gattered_datetime = DateTime()
