import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import langchain_openai as lcai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from agents.constants import databases
from langchain_core.runnables import RunnablePassthrough
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
    database: str = Field(description="databases needed separated by comma")
    request: str = Field(description="data request made to the databases")

class InformationSeekingAgent:
    def __init__(self):
        self.information_chain = self.information_agent()

    def invoke(self, input_params):
        databases_section = "\n".join(
            f"    {idx + 1}. {db_name}  \n"
            f"       - Info: {details['info']}  \n"
            f"       - Device: {details['device']}  \n"
            for idx, (db_name, details) in enumerate(databases.items())
        )

        info_chain = self.information_chain.invoke({'understanding': input_params['understanding'],
                                                    'user_query': input_params['user_query'],
                                                    'hypothesis': input_params['hypothesis'],
                                                    'memory': input_params['memory'],
                                                   'databases_section': databases_section})

        return info_chain

    def information_agent(self):


        prompt = """
    Task: Your role is to create a very specific information request to the databases to answer the QUESTION based on the UNDERSTANDING and ACTION PLAN and MEMORY. 
    UNDERSTANDING might contain partial answer to user query and might have information on what additional data is needed. ACTION PLAN can also help in deciding what data is needed. MEMORY contains all the data already fetched from the databases. 
    
    \Your information request should include any aggregation, comparisons, computation and data processing you need to do to answer the question.
    
    if you are asking for data for more than one day ask aggregated data rather than just raw data.

    Use as many databases as you need to answer the question. 
    
    Always mention user_id in your request. You can get user_id from user_query

    
        "QUESTION": 
          {user_query}
          
          "ACTION PLAN":
          {hypothesis}
          
          "MEMORY": 
          {memory}
      
          "UNDERSTANDING": 
          {understanding}
    
        
   {databases_section}
   
   output format: {{"database": "databases needed", "request": request made to the databases}}
   example: {{"database": "database1, database2", "request": "summarize data for user_id 1234"}}
   
   database name should be taken from information above
   
   your request should be specific having specific time, date, user_id, and other details:
   for example:
    not specific: summarize list of activity and postures for test007 on 07/09/2024
    specific: write code to get the step count, ambient noise, uEMA and heart rate for user_id test007 on 2024-07-09 from 00:00:00 to 23:59:59 from database1, database2 and database3.
    ---      
        
        Instructions:
        2. UNDERSTANDING usually contains information about what kind of data is needed next to answer the question
        
        3. Have date and time in your requests in "%Y-%m-%d %H:%M:%S" format
        
        4. make the request to the database in the natural language from which you need information the most. return request in json format. {{"database name": request}}
        
        5. if you are asking for data for more than one day ask aggregated data rather than just raw data.
                        
        6. Do not ask any questions for which the answer is already present in the UNDERSTANDING.
        
        7. You want all data processing to be done by the database manager, including any filter, aggregation, or transformation, and just want the summarized final result
        
        8. You can return {{"NOT POSSIBLE":""}} as the response if databases don't contain kind of information asked.
        
        9. Make your request in natural language and not in SQL queries. 
        
        10. Make you request in lowercase. Do not capitalize any words in your request.  
           
        11. If understanding says it does not contain a particular kind of data trust it. 

        12. You only create request to get the data. Do not ask for additional processing (NO merging, aggregation, filtering, creating dataframes, doing inferences).

        13. Make a request to use the provided function and print out the results. Do not create any additional functions.

        14. Always have start and end date time in your requests in "%Y-%m-%d %H:%M:%S" format, inside the 'user_query' field. do not create other fields.
        """
        parser = JsonOutputParser(pydantic_object=Output)

        prompt = PromptTemplate(
            template=prompt + '\n {format_instructions}',
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llmchat | parser

        return chain

if __name__ == "__main__":
    question = "summarize day of pilot9 on 2025-03-02?"
    understanding = ""
    memory = ""
    response = InformationSeekingAgent().invoke({'understanding': understanding, 'user_query': question, 'hypothesis': "", 'memory': memory})
    print(response)


