import uuid
from flask import Flask

app = Flask(__name__)
app.secret_key = uuid.uuid4().hex

# Register Blueprints
from main.routes import main_bp
app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True)
