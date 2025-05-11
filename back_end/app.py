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
# MODEL = "meta-llama/llama-4-maverick:free"
MODEL = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Json output schema
class QuestionItem(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(descriptoin="The list of answer choices")
    answer: str = Field(description="The correct answer out of the choices")


class ResponseList(BaseModel):
    questions: List[QuestionItem]

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
        citations = []
        if allow_search and TAVILY_API_KEY:
            search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
            search_results = search_tool.invoke(prompt)  # returns a list of dicts

            context = str(search_results)
            citations = [
                {"title": res["title"], "url": res["url"]}
                for res in search_results
                if "title" in res and "url" in res
            ]

        print('context:', context)

        # # System instructions
        system_prompt = f"""You are a helpful test generator.
        Generate {num_questions} multiple-choice questions with {num_choices} choices each, based on the context below.

        Each item should include:
        - 'question': a clear question
        - 'choices': a list of strings
        - 'answer': the correct answer string

        Each choices' ideas shouldn't overlap with other choices' ideas.
        Make sure that there is only one clear correct answer and no multiple correct answers. For example, if you give a question like 'Define the word bat.', 
        you shouldn't have both choices 'an implement with a handle and a solid surface, usually of wood, used for hitting the ball in games such as baseball, cricket, and table tennis.'
        and 'a mainly nocturnal mammal capable of sustained flight, with membranous wings that extend between the fingers and connecting the forelimbs to the body and the hindlimbs to the tail'
        because they are both correct. Another example is: if you give a question like 'Where did the US drop atomic bombs on Japan in World War 2?', you shouldn't have both choices
        'Nagasaki' and 'Hiroshima' because they are both correct answers. 

        Focus on making questions that allow the test taker to think and apply their knowledge.

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

        raw_response = structured_llm.invoke(messages)
        print('raw_response:', raw_response)

        if raw_response is None:
            raise ValueError("LLM returned None — possibly an invalid response or parsing error.")
        
        questions = raw_response.model_dump()["questions"]

        full_response = {
            "questions": questions,
            "citations": citations
        }
        print('full_response:', full_response)
        return jsonify([full_response])
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"Internal server error {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)