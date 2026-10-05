import os

from langchain_openai import ChatOpenAI

# Chat model shared by all GLOSS4HAR agents
llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     model_name="gpt-4o",
                     temperature=0)
