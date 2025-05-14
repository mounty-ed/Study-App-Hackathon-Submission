from flask import Flask
from routes.prompt_multiple_choice_test import prompt_multiple_choice_test_bp
from routes.upload_multiple_choice_test import upload_multiple_choice_test
from routes.file_upload import file_upload_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])


# Register Blueprints
app.register_blueprint(prompt_multiple_choice_test_bp)
app.register_blueprint(upload_multiple_choice_test)
app.register_blueprint(file_upload_bp)

if __name__ == '__main__':
    app.run(debug=True)

