import os
import sys

import langchain_openai as lcai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.constants import result_expainations
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from agents.data_driver import all_functions

from agents.data_driver import run_function_from_dict, json_to_dict, get_function_description
from agents.constants import USE_AZURE, USE_GPT5, USE_GPT_OSS
from agents.gpt_oss_langchain import GPTOSSChatModel
GPTOSSChatModel.model_rebuild()

if not USE_AZURE and not USE_GPT5 and not USE_GPT_OSS:
    llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                    model_name="gpt-4o",
                    temperature= 0)
elif USE_GPT_OSS:
    llmchat = GPTOSSChatModel()
elif USE_GPT5:
    llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                    model_name="gpt-5")
else:
    llmchat = lcai.AzureChatOpenAI(
                    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                    azure_deployment="PROPILOT",
                    openai_api_version="2024-05-01-preview",
                    model_name="gpt-4o",
                    temperature=0)

class Output(BaseModel):
    understanding: str = Field(description="UNDERSTANDING is the current understanding to answer the user query")


class OutputLocal(BaseModel):
    summary: str = Field(description="result is the summarization of the result")

class OutputNextStep(BaseModel):
    next_step: str = Field(description="returned next step ")


class SenseMakingAgent:
    def __init__(self):
        self.sensemaking_chain = self.sensemaking_agent()
        self.next_step_chain = self.next_step_agent()
        self.global_sensemaking_chain = self.global_sensemaking_agent()

    def invoke_local_sense(self, input_params):
        prompt = self.generate_prompt_local_sense_making(input_params['results'], input_params['data_type'], input_params['user_query'])
        info_chain = self.sensemaking_chain.invoke({"prompt": prompt, "user_query": input_params['user_query']})

        return info_chain

    def invoke_global_sense(self, input_params):
        info_chain = self.global_sensemaking_chain.invoke({'user_query': input_params['user_query'],
                                                           'understanding': input_params['understanding'],
                                                           'memory': input_params['memory'],
                                                           'hypothesis': input_params['hypothesis']})

        return info_chain

    def invoke_next_step(self, input_params):
        next_step_chain = self.next_step_chain.invoke({'user_query': input_params['user_query'],
                                                       'memory': input_params['memory'],
                                                       'understanding': input_params['understanding'],
                                                       'hypothesis': input_params['hypothesis']})

        return next_step_chain

    def generate_prompt_local_sense_making(self, results, data_type, question):
        p = f"Your role is to make sense and summarize the {data_type} data based on user's query \n"
        p += f"User Query: {question} \n\n"
        for r in results:
            function_name = r['func']['name']
            p += f"The results are depicted below for function {r['func']} \n\n"

            p += f"Information about function:\n {get_function_description(all_functions, function_name)} \n\n"

            p += f"Results:\n {r['result']} \n\n"

        p += """
        Instructions:
        1) Summarize the results based on individual sensors.
        1) Use the information in the results to summarize the data based on user's question.
        2) Be clear on what data was used to ger the results.
        3) Return the summarization of result as JSON with key "summary" and value as the summarization.
        """
        return p

    def sensemaking_agent(self):
        prompt = \
            """
            {prompt}
            """

        parser = JsonOutputParser(pydantic_object=OutputLocal)

        prompt = PromptTemplate(
            template=prompt + '\n {format_instructions}',
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = (prompt | llmchat | parser)

        # chain = (template | llmchat| StrOutputParser())

        return chain

    def next_step_agent(self):
        prompt = \
            """
            Your task is to determine the next step based on the information provided in "understanding" and "User Query"
            
            HYPOTHESIS:
            {hypothesis}
        
            UNDERSTANDING:
            {understanding}
            
            USER QUERY:
            {user_query}
            
            
            Next Step Options:
            - INF: More data is needed to answer the "USER QUERY" based on "HYPOTHESIS" and "UNDERSTANDING"
            - END: The the UNDERSTANDING fully answers the "USER QUERY" based on "HYPOTHESIS"
            - END: If the "understanding" contains CODE-999.
        
             Do not return {{"next_step": "END"}} if the "UNDERSTANDING" does not have a satisfactory answer to the "USER QUERY"
            
            
            Instructions:
            1) Return {{"next_step": "INF"}} if the "UNDERSTANDING" indicates additional information is needed.
            2) Do not {{"next_step": "END"}} if "UNDERSTANDING" indicates that information from other databases can be used for additional verification.
            4) Return {{"next_step": "END"}} if the "understanding" fully answers the "USER QUERY" or contains CODE-999.
            5) return {{"next_step": "INF"}} or {{"next_step": "END"}}
            
            Example: 
            {{"next_step": "INF"}}  
            {{"next_step": "END"}}
            """
        parser = JsonOutputParser(pydantic_object=OutputNextStep)

        prompt = PromptTemplate(
            template=prompt + '\n {format_instructions}',
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = (prompt | llmchat | parser)

        return chain

    def global_sensemaking_agent(self):
        prompt = \
            """
            USER QUERY: 
            {user_query}
            
            HYPOTHESIS:
            {hypothesis}
             
            MEMORY: 
            {memory}
            
            UNDERSTANDING: 
            {understanding}
            
            USER QUERY is the question asked by the user.
            
            HYPOTHESIS is the possible hypothesis generated to answer the USER QUERY

            UNDERSTANDING is the current answer the USER QUERY
            
            MEMORY is all the data gathered so far
            
            
            Your role is to rewrite the UNDERSTANDING such that it answers USER QUERY based on the information in MEMORY, current UNDERSTANDING and HYPOTHESIS.
            If the more data is needed to answer the query which is not available in MEMORY yet, then include the kind of data needed in the UNDERSTANDING.
            
            
            UNDERSTANDING should include ALL the important information from Memory that can be needed later to answer the USER QUERY. 
            Include any important information in Memory revelant to the USER QUERY. Do not lose information.
            Do not suggest additional data if current data sufficiently answers the USER QUERY.
        
            
            
            Instructions:
            1) Update the understanding based on the information in Memory and User Query if required.
            2) Use common sense to create an understanding.
            3) Update the understanding based on individual sensors and their results.
            3) Just return updated understanding in Format {{"understanding": "updated understanding"}}
            4) Do not have ```json or ```python in your answer
            5) The understanding should be in natural language and dates in understanding should be in "%Y-%m-%d %H:%M:%S" format.
            6) if memory has CODE-999 include that in understanding
            7) You should resynthesize UNDERSTANDING such that it answers the USER QUERY best based on current data and also contains information on additional data needed.
            8) if some data was tried fetching from memory and was not found, include that in understanding.
            """
        # template = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", prompt)
        #     ]
        # )

        parser = JsonOutputParser(pydantic_object=Output)

        prompt = PromptTemplate(
            template=prompt + '\n {format_instructions}',
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = (prompt | llmchat | parser)

        return chain


if __name__ == "__main__":
    question = (
        "can you provide location of places where pilot5 was stationary based 2025-03-02 data")
    response = SenseMakingAgent().make_sense({'user_query': question, 'memory': '', 'understanding': "", "hypothesis": ""})
    print(response)
