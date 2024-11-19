from config import DB, CustomDateTime
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy import Uuid, String, Text, Enum, DateTime, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from .behavior import Behavior
from .recommendation_log import Recommendationlog

class Item(DB.Model):
    __tablename__ = 'item'

    uuid = mapped_column(Uuid(as_uuid=True), primary_key=True)
    title = mapped_column(String(500), nullable=False)
    category = mapped_column(String(100), nullable=True)
    abstract = mapped_column(Text, nullable=False)
    link = mapped_column(String(255))
    data_source = mapped_column(Enum('mind_small',
                                     'hf_paper',
                                     'cnn_news',
                                     'brief_ai_news',
                                     'tech_news',
                                     'bbc_news',
                                     'mit_news'), nullable=False)
    gattered_datetime = mapped_column(DateTime)

    behavior = relationship('Behavior', backref='item', uselist=True, lazy='dynamic')
    recommendationlog = relationship('Recommendationlog', backref='item', uselist=True, lazy='dynamic')

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

class ItemSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Item
        load_instance = False # load as SQLAlchemy instance or plain dict.
    gattered_datetime = CustomDateTime()
