
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from agents.constants import ABLATION_PRESENTATION_AGENT
from agents.llm import llmchat


class Output(BaseModel):
    response: str = Field(description="response to the user query based on instructions")


class PresentationAgent:
    def __init__(self):
        self.presentation_chain = self.presentation_chain()

    def invoke(self, input_params):
        presentation_chain = self.presentation_chain.invoke(
            {'user_query': input_params['user_query'], 'understanding': input_params['understanding'],
             'instructions': input_params['instructions']})

        return presentation_chain

    def presentation_chain(self):

        if ABLATION_PRESENTATION_AGENT:
            prompt = """    
            Task: Given the user query, the understanding of the query, and instructiosn, your role is to provide the answer to the user query according to the instructions provided.
            
            USER QUERY:
            {user_query}
            
            UNDERSTANDING:
            {understanding}
            
            INSTRUCTIONS:
            {instructions}
            
            Output Instructions:
            1. Use INSTRUCTIONS to provide the answer to the user query.
            2. Do not include CODE-999 in response
            3. Present time in more format: HH:MM
            4. Your answer format is list of: [start time]-[end time]: posture: [posture]; activities: [activities]; reasoning: [reasoning].
        """
        else:
            prompt = """    
                Task: Given the user query, the understanding of the query, and instructiosn, your role is to provide the answer to the user query according to the instructions provided.
                
                USER QUERY:
                {user_query}
                
                UNDERSTANDING:
                {understanding}
                
                INSTRUCTIONS:
                {instructions}
                
                Output Instructions:
                1. Use INSTRUCTIONS to provide the answer to the user query.
                2. Do not include CODE-999 in response
                3. Present time in more format: HH:MM
                4. Your answer format is list of: [start time]-[end time]: posture: [posture]; activities: [activities]; reasoning: [reasoning].
                6. Only use the postures among this list: 'sitting', 'standing', 'lying down', 'reclining', 'upright'
                7. Only use the activities among this list: 'video gaming',  'walking', 'stair climbing',  'getting ready',  'driving',  'bicycling',  'vigorous bicycling',  'aerobics',  'cleaning',  'cooking',  'laundry',  'playing with pet',  'listening to music',  'watching movies/TV',  'studying',  'reading',  'riding car',  'riding train',  'riding bus',  'playing musical instruments',  'attending meeting',  'computer using', 'phone using',  'running',  'getting dressed',  'grooming',  'using bathroom',  'eating',  'talking','strength training',  'washing dishes',  'carrying groceries',  'putting away groceries',  'shopping',  'making bed',  'packing/unpacking',  'sleeping',  'playing sports'
                8 Choose the closert activity or posture from the list above if the activity is not in the list.
                9. Never include any other activity or posture that is not in the list above.

                Example response:
                - 10:00-10:03: posture: sitting; activities: computer using, reading; reasoning: Phone used and no steps detected, so likely sitting and using computer.
                - 10:03-11:12: posture: standing; activities: walking, phone using; reasoning: Detected steps and outdoors noise, so likely standing and walking.
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
    question = "Did test 004 walked without carrying their phone but wearing the garmin watch on 07/20/24 6pm - 9pm."
    instructions = '''
    Answer the question and include your reasoning in the response.
    answer: YOUR ANSWER  
    reasoning: INCLUDE YOUR REASONING for the answer
    '''
    response = PresentationAgent().invoke({'user_query': "find the name of wifi that is most likely home wifi of test004 was on 07/09/2024?",
                                           'understanding': "On 2024-07-09, user 'test004' spent a total of 21.47 hours connected to the 'FeelTheConnection' Wi-Fi network, 1.95 hours connected to the 'NUwave' Wi-Fi network, and approximately 0.55 hours not connected to any Wi-Fi. The Wi-Fi network 'FeelTheConnection' was the one the phone was connected to for the longest duration, indicating it is most likely the home Wi-Fi network.",
                                           'instructions': instructions})

    print(response)
