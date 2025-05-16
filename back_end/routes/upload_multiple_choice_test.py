from flask import Blueprint, request, jsonify
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
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


upload_multiple_choice_test = Blueprint('upload_multiple_choice_test', __name__)

# Global config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL = "openai/gpt-4o-mini"
MODEL = "google/gemini-2.0-flash-001"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
UPLOAD_FOLDER = 'uploads'

# Json output schema
class QuestionItem(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(description="The list of answer choices")
    answer: str = Field(description="The correct answer out of the choices")
    explanation: str = Field(description="The explanation for the correct answer")


class ResponseList(BaseModel):
    questions: List[QuestionItem]


@upload_multiple_choice_test.route('/api/generate-upload-multiple-choice-test', methods=['POST'])
def generate_upload_multiple_choice_test():
    # --------------------------------------------------------------------- LOAD VARIABLES ---------------------------------------------------------------------

    if not OPENROUTER_API_KEY:
        return jsonify({"error": "Missing OPENROUTER_API_KEY"}), 500

    data = request.get_json()
    prompt = data.get("prompt")
    num_questions = data.get("numQuestions")
    num_choices = data.get("numChoices")
    file_id = data.get('file_id')

    print('data:', data)

    if not all([prompt, num_questions, num_choices, file_id]):
        return jsonify({"error": "Missing required fields"}), 400

    filepath = None
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.startswith(file_id):
            filepath = os.path.join(UPLOAD_FOLDER, fname)
            break
    if not filepath:
        return jsonify({'error': 'File not found'}), 404

    try:
        # --------------------------------------------------------------------- RAG SETUP ---------------------------------------------------------------------
        # Load, split, and embed PDF
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        for i, doc in enumerate(documents):
            doc.metadata["source"] = f"{fname} - Page {i + 1}"

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

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
            {"messages":[{"role":"user","content": prompt}]},
            config={"recursion_limit": 50}
        )

        result: ResponseList = output["structured_response"]
        print(result.model_dump())
        return jsonify(result.model_dump()), 200



    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
