from flask import Blueprint, request, jsonify, current_app
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import mimetypes
import re
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredURLLoader,
    YoutubeLoader
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import initialize_agent, AgentExecutor, tool
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
import requests
import tempfile


link_multiple_choice_test_bp = Blueprint('link_multiple_choice_test', __name__)


# Json output schema
class QuestionItem(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(descriptoin="The list of answer choices")
    answer: str = Field(description="The correct answer out of the choices")
    explanation: str = Field(description="The explanation for the correct answer")

class ResponseList(BaseModel):
    questions: List[QuestionItem]


@link_multiple_choice_test_bp.route('/api/generate-link-multiple-choice-test', methods=['POST'])
def generate_link_multiple_choice_test():
    # Global config
    EMBEDDING_MODEL = current_app.config['EMBEDDING_MODEL']
    MODEL = current_app.config['MODEL']
    OPENROUTER_API_KEY = current_app.config['OPENROUTER_API_KEY']
    UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']
    # --------------------------------------------------------------------- LOAD VARIABLES ---------------------------------------------------------------------

    if not OPENROUTER_API_KEY:
        return jsonify({"error": "Missing OPENROUTER_API_KEY"}), 500

    data = request.get_json()
    prompt = data.get("prompt")
    num_questions = data.get("numQuestions")
    num_choices = data.get("numChoices")
    link = data.get('link')

    print('data:', data)

    if not all([num_questions, num_choices, link]):
        return jsonify({"error": "Missing required fields"}), 400


    try:
        # --------------------------------------------------------------------- RAG SETUP ---------------------------------------------------------------------
        def is_youtube(link):
            return "youtube.com" in link or "youtu.be" in link

        def is_article(link):
            return link.startswith("http") and not is_youtube(link)
        

        def load_link(link):
            # YouTube Link
            if is_youtube(link):
                print('youtube link identified')
                loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)
                return loader.load()

            # Web Article
            if is_article(link):
                print('article link identified')
                loader = UnstructuredURLLoader(urls=[link])
                return loader.load()


            print('file link identified')
            # Direct File Link (.pdf, .txt, .doc, .docx)
            response = requests.get(link)
            if response.status_code != 200:
                raise ValueError("Failed to download file from URL")

            content_type = response.headers.get("Content-Type", "")
            ext = mimetypes.guess_extension(content_type) or os.path.splitext(link)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(response.content)
                filepath = tmp_file.name

            # Load based on extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(filepath)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            return loader.load()

        # Load the document
        documents = load_link(link)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        # Define tool for the agent
        @tool
        def retrieve_context(query: str) -> str:
            """Use this to retrieve relevant context for test generation from the uploaded document."""
            docs = retriever.invoke(query)
            print("\n\n".join([doc.page_content for doc in docs]))
            return "\n\n".join([doc.page_content for doc in docs])
        

        # --------------------------------------------------------------------- Agent Initialization ---------------------------------------------------------------------     

        parser = PydanticOutputParser(pydantic_object=ResponseList)
        format_instructions = parser.get_format_instructions()
   
        print('starting agent initialization')

        llm = ChatOpenAI(
            model=MODEL,
            temperature=0.1,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # build prompt with format instructions
        system_prompt = f"""
        You are a helpful multiple-choice test generator. Use retrieve_context() to fetch document excerpts.
        The user will describe a test they want, and you will:
        1. Call the tool to get context based on the topic.
        2. Generate exactly {num_questions} multiple-choice questions with {num_choices} answer choices each, based on that context.
        3. **Strictly** output JSON matching the given schema.

        {format_instructions}

        Ensure that:
        - Choices are mutually exclusive.
        - Only one correct answer is provided.
        - Each question is unique from one another. NO DUPLICATES.
        - The test follows any specifications in the prompt and covers a wide variety within those specifications.
        - Each explanation is brief, but useful and educational. No more than 2 sentences.
       
        Call the tool multiple times if necessary.

        When you are completely done generating all questions, immediately output **only** the final JSON object (no thoughts, no tool calls, no extra text). That final JSON is your last token output.
        """

        agent = create_react_agent(
            model=llm,
            tools=[retrieve_context],
            prompt=system_prompt.strip(),
            response_format=(system_prompt.strip(), ResponseList),
        )

        # invoke and parse
        output = agent.invoke(
            {"messages":[{"role":"user","content": f"Generate a multiple-choice test based on the uploaded document. Additional Directions: {f'{prompt}' if prompt else "none"}"}]},
            config={"recursion_limit": 50}
        )

        result: ResponseList = output["structured_response"]
        print(result.model_dump())
        return jsonify(result.model_dump()), 200



    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
