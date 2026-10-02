import os

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from agents.constants import USE_AZURE, USE_GPT5
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

class Output(BaseModel):
    hypothesis_1: str = Field(description="first hypothesis")
    hypothesis_2: str = Field(description="second hypothesis")
    hypothesis_3: str = Field(description="third hypothesis")


class HypothesisGeneratorAgent:
    def __init__(self):
        self.information_chain = self.information_agent()

    def invoke(self, input_params):
        info_chain = self.information_chain.invoke({'user_query': input_params['user_query']})

        return info_chain

    def information_agent(self):
        prompt = """
    Task: Given the available databases and the user's query, generate up to three hypotheses on how the user's question can be answered using the available databases. 
    You can use multiple databases in your hypothesis. Your hypothesis should try to answer all parts of the user query.

    ---
    User Query:
    {user_query}
    
    Databases Available:
    
    1. activity database  
       - Info: Contains activity data (e.g., "stationary," "automotive," "cycling," "walking," "running") recorded via accelerometer and gyroscope sensors in the phone. If the phone is not carried, the user is assumed inactive.  
       - Device: Phone  
    
    2. location database  
       - Info: Contains latitude, longitude, and altitude data per minute, along with functions to fetch address and calculate metrics like time spent at a location and location paths.  
       - Device: Phone  
    
    3. phone steps database  
       - Info: Contains steps walked, floors ascended, floors descended, and distance covered between two intervals, calculated via the phone.  
       - Device: Phone  
    
    4. garmin steps database  
       - Info: Contains steps walked between two intervals, calculated via Garmin smartwatch. Measures the same thing as phone steps database but using garmin smart watch.
       - Device: Garmin Smartwatch  
    
    5. garmin hr database  
       - Info: Contains heart rate data (per 30 seconds) and functions for summarizing heart rate data recorded from the Garmin smartwatch.  
       - Device: Garmin Smartwatch  
    
    6. lock unlock database  
       - Info: Contains records of phone lock and unlock times, with functions to extract this information.  
       - Device: Phone 
       
    7. wifi database
         - Info: Contains data whether phone is connected to wifi or not. It also contains the wifi name to which phone is connected.
         - Device: Phone 

    ---

    Instructions:
    
    1. Based on the user's query, generate up to three hypotheses on how the query might be answered using the available databases.
       - Format the hypotheses as: "hypothesis 1": hypothesis 1, "hypothesis 2": hypothesis 2, "hypothesis 3": hypothesis 3.
       - Rank hypotheses by strength, listing the strongest first.
       
    2. If the answer cannot be directly determined, provide approximations where possible.
    
    3. Each hypothesis should reference the database(s) and describe how the information can be used to answer the query.
    
    4. Ensure the output is formatted in JSON, with natural language used for hypotheses.
    
    5. Return the any date and time in the format "%Y-%m-%d %H:%M:%S".
    
    6. Use Common Sense and your world knowledge to generate the hypotheses.
    
    7. Limit: Generate a maximum of three hypotheses. If questions is very direct and can just be answer by a single hypothesis just return that hypothesis.
    
    8. If the question cannot be answered with the given data. The hypothesis should say that.
    
    9. Your hypothesis should be independent of other hypothesis. For example hypothesis 1 should not depend on hypothesis 2.
    10. You can use multiple databases in your hypothesis if needed to answer the user query.
    """
        parser = JsonOutputParser(pydantic_object=Output)

        # template = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", prompt),
        #     ]
        # )
        prompt = PromptTemplate(
            template=prompt + '\n {format_instructions}',
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llmchat | parser

        return chain


if __name__ == "__main__":
    question = "Use all databases to provide an understanding of what test004 was doing at 3:40 AM?"
    response = HypothesisGeneratorAgent().invoke({'user_query': question})
    print(response)
