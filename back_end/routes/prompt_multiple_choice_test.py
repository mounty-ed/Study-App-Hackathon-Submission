from flask import Blueprint, request, jsonify, current_app
import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
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

prompt_multiple_choice_test_bp = Blueprint('prompt_multiple_choice_test', __name__)


# Json output schema
class QuestionItem(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(descriptoin="The list of answer choices")
    answer: str = Field(description="The correct answer out of the choices")
    explanation: str = Field(description="The explanation for the correct answer")

class ResponseList(BaseModel):
    questions: List[QuestionItem]

@prompt_multiple_choice_test_bp.route("/api/generate-prompt-multiple-choice-test", methods=["POST"])
def generate_prompt_multiple_choice_test():
    # Global config
    EMBEDDING_MODEL = current_app.config['EMBEDDING_MODEL']
    MODEL = current_app.config['MODEL']
    OPENROUTER_API_KEY = current_app.config['OPENROUTER_API_KEY']
    TAVILY_API_KEY = current_app.config['TAVILY_API_KEY']
    # --------------------------------------------------------------------- LOAD VARIABLES ---------------------------------------------------------------------

    if not OPENROUTER_API_KEY:
        return jsonify({"error": "Missing OPENROUTER_API_KEY"}), 500
    
    data = request.get_json()
    prompt = data.get("prompt")
    num_questions = data.get("numQuestions")
    num_choices = data.get("numChoices")
    allow_search = bool(data.get("allowSearch", False))

    print("Incoming data:", data)

    if not prompt or not num_questions or not num_choices:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        # --------------------------------------------------------------------- WEB SEARCH TOOL SETUP ---------------------------------------------------------------------
        # Handle optional search
        citations = []
        search_tool = TavilySearchResults(
            api_key=TAVILY_API_KEY, 
            max_results=6, 
            search_depth='advanced',
            include_domains=["wikipedia.org", "britannica.com"])

        @tool
        def retrieve_context(query: str) -> str:
            """Use this to retrieve relevant context on the web for test generation."""
            search_results = search_tool.invoke(query)
            citation = [
                {"title": res["title"], "url": res["url"]}
                for res in search_results
                if "title" in res and "url" in res
            ]
            citations.extend(citation)
            return "\n\n".join(
                f"{res.get('title', '')}\n{res.get('content', '')}"
                for res in search_results
                if "content" in res
            )

        # --------------------------------------------------------------------- Agent Initialization ---------------------------------------------------------------------     

        parser = PydanticOutputParser(pydantic_object=ResponseList)
        format_instructions = parser.get_format_instructions()

        print('starting agent initialization')

        if allow_search:
            system_prompt = f"""
            You are a helpful multiple-choice test generator. Use retrieve_context() to fetch online web information.
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

            Call the tool multiple times if necessary.

            When you are completely done generating all questions, immediately output **only** the final JSON object (no thoughts, no tool calls, no extra text). That final JSON is your last token output.
            """
        else:
            system_prompt = f"""
            You are a helpful multiple-choice test generator.
            The user will describe a test they want, and you will:
            1. Generate exactly {num_questions} multiple-choice questions with {num_choices} answer choices each, based on that context.
            2. **Strictly** output JSON matching the given schema.

            {format_instructions}

            Ensure that:
            - Choices are mutually exclusive.
            - Only one correct answer is provided.
            - Each question is unique from one another.
            - The test follows any specifications in the prompt and covers a wide variety within those specifications.
            - Each explanation is brief, but useful and educational. No more than 2 sentences.

            Call the tool multiple times if necessary.

            When you are completely done generating all questions, immediately output **only** the final JSON object (no thoughts, no tool calls, no extra text). That final JSON is your last token output.
            """

        llm = ChatOpenAI(
            model=MODEL,
            temperature=0.4,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            request_timeout=60,
        )
        
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

        print('raw output:', output)

        result: ResponseList = output["structured_response"]
        questions = [q.model_dump() for q in result.questions]

        full_response = {
            "questions": questions,
            "citations": citations
        }
        print('full_response:', full_response)
        return jsonify([full_response]), 200
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"Internal server error {str(e)}"}), 500
