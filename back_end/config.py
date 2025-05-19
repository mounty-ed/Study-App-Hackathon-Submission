import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'storage', 'document_uploads')
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL = "openai/gpt-4o-mini"
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")




