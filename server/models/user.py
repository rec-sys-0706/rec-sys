from config import DB
from sqlalchemy.orm import mapped_column
from sqlalchemy import VARCHAR, Uuid, String, inspect
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class User(DB.Model):
    __tablename__ = 'app_user'

    uuid = mapped_column(Uuid(as_uuid=True), primary_key=True)
    account = mapped_column(String(64), nullable=False)
    password = mapped_column(VARCHAR(256), nullable=False)
    email = mapped_column(String(128), nullable=False)
    line_id = mapped_column(String(64))

    def serialize(self) -> dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
    
class UserSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = User
        load_instance = False