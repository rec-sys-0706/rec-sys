from config import DB
from sqlalchemy.orm import mapped_column
from sqlalchemy import UUID, String, Text, Enum, DateTime, inspect

class Item(DB.Model):
    uuid = mapped_column(UUID(as_uuid=True), primary_key=True)
    title = mapped_column(String(500), nullable=False)
    abstract = mapped_column(Text, nullable=False)
    link = mapped_column(String(255))
    data_source = mapped_column(Enum('mind_small',
                                     'hf_paper',
                                     'cnn_news'), nullable=False)
    gattered_datetime = mapped_column(DateTime)

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

