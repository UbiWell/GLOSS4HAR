import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from agents.constants import databases
from agents.llm import llmchat


class Output(BaseModel):
    action_plan: str = Field(description="action plan")


class HypothesisGeneratorAgentAlt1:
    def __init__(self):
        self.information_chain = self.information_agent()

    def generate_prompt(self, user_query, databases):
        """
        Generates a formatted prompt string for the task based on the given user query and databases.

        Parameters:
        - user_query (str): The user query.
        - databases (dict): A dictionary where each key is a database name and each value is a dictionary with 'info' and 'device'.

        Returns:
        - str: The formatted prompt.
        """

        # Formatting the databases section
        databases_section = "\n".join(
            f"    {idx + 1}. {db_name}  \n"
            f"       - Info: {details['info']}  \n"
            f"       - Device: {details['device']}  \n"
            + (f"       - Additional Instructions: {details['additional_instructions']}  \n"
               if 'additional_instructions' in details else '')
            for idx, (db_name, details) in enumerate(databases.items())
        )

        # Defining the prompt template
        prompt_template = f"""
    Task: Given the available databases and the user's query, generate up to an action plan on how the user's question can be answered using the available databases. 
    You can use multiple databases in your action plan. Your action plan should try to answer all parts of the user query.

        ---
        User Query:
        {user_query}

        Databases Available:

        {databases_section}

        ---
        
        Additonal Instructions:
        1) Garmin steps 

         Instructions:

    1. Based on the user's query, generate up an action plan on how the query might be answered using the available databases.

    2. Your action plan should assume that all the processing happens by database functions. 

    2. If the answer cannot be directly determined, your action plan should provide approximations where possible.

    3. The action plan should reference the database(s) and describe how the information can be used to answer the query.

    4. Ensure the output is formatted in JSON, with natural language used for action plan. All the steps of action plan should be included in same string and return a JSON {{"action_plan": your plan}}.

    5. Return the any date and time in the format "%Y-%m-%d %H:%M:%S". If ambiguous, assume date in query is in %Y-%m-%d %H:%M:%S" format and mention this in your action plan.

    6. Use Common Sense and your world knowledge to generate the action plan.

    8. If the question cannot be answered with the given data. The action plan should just say that "The query cannot be answered with given datasets".

    9. You can use multiple databases in your action plan only when needed. If you can answer the query with single database, you should use only that database.
        """

        return prompt_template.strip()

    def invoke(self, input_params):
        prompt = self.generate_prompt(input_params['user_query'], databases)
        info_chain = self.information_chain.invoke({'user_query': input_params['user_query'], 'prompt': prompt})

        return info_chain

    def information_agent(self):
        prompt = """ {prompt}"""

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
    question = "what are opportune times of drinking water for test004 on 07/09/2024?"
    response = HypothesisGeneratorAgentAlt1().invoke({'user_query': question})
    print(response['action_plan'])
