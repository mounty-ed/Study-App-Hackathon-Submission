from flask import Flask
from routes.prompt_multiple_choice_test import prompt_multiple_choice_test_bp
from routes.upload_multiple_choice_test import upload_multiple_choice_test
from routes.link_multiple_choice_test import link_multiple_choice_test_bp
from routes.file_upload import file_upload_bp
from routes.upload_flashcards import upload_flashcards_bp
from routes.link_flashcards import link_flashcards_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

app.register_blueprint(prompt_multiple_choice_test_bp)
app.register_blueprint(upload_multiple_choice_test)
app.register_blueprint(file_upload_bp)
app.register_blueprint(upload_flashcards_bp)
app.register_blueprint(link_flashcards_bp)
app.register_blueprint(link_multiple_choice_test_bp)

if __name__ == '__main__':
    app.run(debug=True)

