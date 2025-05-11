from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from flask_cors import CORS
import os

MODEL = "meta-llama/llama-4-maverick:free"
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

class QuestionItem(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(descriptoin="The list of answer choices")
    answer: str = Field(description="The correct answer out of the choices")


class ResponseList(BaseModel):
    questions: List[QuestionItem]

# LLM setup
llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    request_timeout=60,
)

# System prompt for generating questions
system_prompt = """
You are a helpful test generator.
Generate 5 multiple-choice questions with 4 choices each, based on the context below.
Each item must include:
- "question": a clear question
- "choices": a list of 4 strings
- "answer": one of the items from the choices list

Only return a valid JSON object that matches this structure:
{
  "questions": [
    {
      "question": "...",
      "choices": ["...", "...", "..."],
      "answer": "..."
    },
    ...
  ]
}

Context:
Use your general knowledge.
"""

# Test with structured output
# structured_llm = llm.with_structured_output(schema=ResponseList)

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="What are the important aspects of ecosystem services?")
]

# Invoke the LLM and capture the response
response = llm.invoke(messages)
print("Response:", response)
