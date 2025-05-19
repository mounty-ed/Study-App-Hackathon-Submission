# Study Buddy

An AI-powered tool that generates multiple-choice tests and flashcards from any input — PDFs, prompts, articles, or videos.

## Features

- Upload documents, enter text, or insert url
- Generates high-quality tests & flashcards
- Web search and video/article transcription
- Clean Material UI frontend with tabbed navigation
- Test history (coming soon)

## 🛠️ Tech Stack

**Frontend:**  
- Next.js  
- Material UI  

**Backend:**  
- Flask API  
- LangChain for RAG  
- HuggingFace Transformers for embeddings  
- FAISS for vector search  
- Tavily Web Search API  
- OpenRouter for LLM generation  
- Pydantic for structured JSON validation  

## 📦 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/mounty-ed/study-app.git
cd study-buddy
```

### 2. Set up the backend (Flask)
```bash
cd back_end
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd ..
python -m back_end.app
```
The Flask backend should now be running on http://localhost:5000.

### 3. Set up the frontend (Next.js)
```bash
cd ../front_end/my-app
npm install
npm run dev
```
The frontend should now be available at http://localhost:3000.


## Coming Soon
- User authentication

- Study mode with spaced repetition

- Audio/video-based note-taking

- Auto course generation from syllabus


## License
MIT License

Copyright (c) 2025 mounty-ed