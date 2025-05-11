from flask import Flask, request, jsonify
import os
import json
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from flask_cors import CORS

# Flask app

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Declare global variables

MODEL = "qwen/qwen2.5-vl-32b-instruct:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Json output schema

class QuestionItem(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(descriptoin="The list of answer choices")
    answer: str = Field(description="The correct answer out of the choices")

class CitationItem(BaseModel):
    title: str = Field(description="Title of source or citation")
    citation: str = Field(description="Sources or citations for the questions")

class ResponseList(BaseModel):
    questions: List[QuestionItem]
    citations: List[str] = Field(description="Sources or citations for the questions")

@app.route("/api/generate-multiple-choice-test", methods=["POST"])
def generate_multiple_choice_test():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "Missing OPENROUTER_API_KEY"}), 500


    # Get data from body 
    data = request.get_json()
    prompt = data.get("prompt")
    num_questions = data.get("numQuestions")
    num_choices = data.get("numChoices")
    allow_search = bool(data.get("allowSearch", False))

    if not prompt or not num_questions or not num_choices:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # llm config and declaration
        llm = ChatOpenAI(
            model=MODEL,
            temperature=0,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            request_timeout=60,
        )

        # Handle optional search
        context = ""
        if allow_search and TAVILY_API_KEY:
            search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)
            context = search_tool.run(prompt)

        print('context:', context)

        # # System instructions
        system_prompt = f"""You are a helpful test generator.
        Generate {num_questions} multiple-choice questions with {num_choices} choices each, based on the context below.
        Each item should include:
        - 'question': a clear question
        - 'choices': a list of strings
        - 'answer': the correct answer string

        And there should be another item outside the question list:
        - 'citations': the source of the information used for the questions

        Only return a valid JSON array that matches this structure. Do not explain anything else.

        Context:
        {context if context else 'Use your general knowledge'}
        """
        
        # Bind schema to LLM
        structured_llm = llm.with_structured_output(schema=ResponseList)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        response = structured_llm.invoke(messages)
        print('response:', response)
        return jsonify([response.model_dump()])

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)