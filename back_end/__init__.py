import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config.from_object('back_end.config.Config')

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Enable CORS
    CORS(app, origins=["http://localhost:3000"])

    # Register blueprints
    from back_end.routes.prompt_multiple_choice_test import prompt_multiple_choice_test_bp
    from back_end.routes.upload_multiple_choice_test import upload_multiple_choice_test
    from back_end.routes.file_upload import file_upload_bp
    from back_end.routes.upload_flashcards import upload_flashcards_bp
    from back_end.routes.link_flashcards import link_flashcards_bp
    from back_end.routes.link_multiple_choice_test import link_multiple_choice_test_bp

    app.register_blueprint(prompt_multiple_choice_test_bp)
    app.register_blueprint(upload_multiple_choice_test)
    app.register_blueprint(file_upload_bp)
    app.register_blueprint(upload_flashcards_bp)
    app.register_blueprint(link_flashcards_bp)
    app.register_blueprint(link_multiple_choice_test_bp)

    return app
