import os
import sys
import json
import time

import agents.sensemaking_agent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))

from agents import sensemaking_agent, information_seeking_agent, \
    hypothesis_generator_agent_alt_1,generic_database_manager, presentation_agent

is_action_plan = True
max_iters = 1


class SenseMaker:
    def __init__(self, question, presentation_instructions):
        self.user_query = question
        self.presentation_instructions = presentation_instructions
        self.memory = ''
        self.answer = ''
        self.current_step = ""
        self.understanding = ''
        self.hypothesis = ''
        self.function_calls = []
        self.information_request = []
        self.step_history = []
        self.sense_making_agent = agents.sensemaking_agent.SenseMakingAgent()
        self.information_seeking_agent = agents.information_seeking_agent.InformationSeekingAgent()
        self.generic_db_manager = agents.generic_database_manager.GenericDatabaseManager()
        self.presentation_agent = agents.presentation_agent.PresentationAgent()
        self.state_dict = {"INF": "INFORMATION SEEKING", "END": "END"}

        self.hypothesis_generator_agent = agents.hypothesis_generator_agent_alt_1.HypothesisGeneratorAgentAlt1()

    def make_sense(self):
        self.current_step = "START"
        self.step_history.append(self.current_step)
        time.sleep(1)

        if self.user_query == "" or self.presentation_instructions == "":
            self.answer = "Incomplete query or instructions"
            self.current_step = "FINISH"
            return
        
        # print out the user query and instructions
        print(f"User query: {self.user_query}")

        self.current_step = "ACTION PLAN GENERATION"
        self.step_history.append(self.current_step)

        print("Action plan generation step")
        hypos = self.invoke_with_retry(self.hypothesis_generator_agent, 'invoke', {'user_query': self.user_query})

        if "NOT POSSIBLE" in hypos:
            print("Not possible to answer the question with current data")
        else:
            print("Hypothesis generation step")

            if is_action_plan:
                print(f"Action plan: {hypos['action_plan']}")
                self.hypothesis = hypos["action_plan"]
                if (self.hypothesis == "The query cannot be answered with given datasets"):
                    self.answer = "The query cannot be answered with given datasets"
                    self.current_step = "FINISH"
                    return

            else:
                for h in hypos:
                    print(f"{h}: {hypos[h]}")
                choice = '1'
                self.hypothesis = hypos["hypothesis_" + choice]

        num_iters = 0

        while self.current_step != "END":
            print("***************************  MEMORY  *******************************")
            print(self.memory)

            print("***************************  UNDERSTANDING  *******************************")
            print(self.understanding)

            response = self.invoke_with_retry(self.sense_making_agent, 'invoke_next_step', {
                'user_query': self.user_query,
                'memory': self.memory,
                'understanding': self.understanding,
                'hypothesis': self.hypothesis
            })
            try:
                # response_dict = json.loads(response)
                if "next_step" in response:
                    state = response["next_step"]
            except Exception as e:
                print(f"Error in response: {str(e)}")
                continue
            print("Returned state: ", state)

            self.current_step = self.state_dict[state]
            self.step_history.append(self.current_step)

            if state == "INF" or state == 'INF':
                state = "INF"
                print("Information seeking step")
                response = self.invoke_with_retry(self.information_seeking_agent, 'invoke', {
                    'understanding': self.understanding,
                    'user_query': self.user_query,
                    'hypothesis': self.hypothesis,
                    'memory': self.memory
                })

                if 'NOT POSSIBLE' in response:
                    self.memory += f"\n\nNot possible to answer {self.user_query} using the available data. CODE-999."
                    self.understanding += f"\n\nNot possible to answer {self.user_query} using the available data. CODE-999."
                    num_iters += 1
                    continue
                

                if not "database" in response:
                    continue
                if response["database"] == ["NOT_POSSIBLE"]:
                    num_iters += 1
                    continue
                database = response["database"]
                request = response["request"]
                database = database.split(',')
                database = [d.strip() for d in database]
                self.information_request.append(f"{database}: {request}")

                results = None
                # db_manager_map = {
                #     "activity database": self.activity_db_manager,
                #     "location database": self.location_db_manager,
                #     "phone steps database": self.phone_steps_db_manager,
                #     "garmin hr database": self.garmin_hr_db_manager,
                #     "lock unlock database": self.lock_unlock_db_manager,
                #     "garmin steps database": self.garmin_steps_db_manager,
                #     "wifi database": self.wifi_db_manager,
                #     "app usage database": self.app_usage_db_manager,
                #     "phone battery database": self.phone_battery_db_manager,
                #     "call log database": self.call_log_db_manager,
                #     "garmin stress database": self.stress_db_manager
                #
                # }


                results = self.invoke_with_retry(self.generic_db_manager, 'invoke', {'user_query': request, 'databases': database})
                if results == "FAILED":
                    results = None

                elif database == "NOT POSSIBLE":
                    self.memory += f"\n\nNot possible to answer {request} using the available data. CODE-999."

                if results:
                    if ("NOT POSSIBLE" in results):
                        local_sense = f"Failed to generate results because {results['NOT POSSIBLE']}"
                    else:
                        for res in results:
                            if ('func' in res):
                                self.function_calls.append(res['func'])
                        print("local sense making step")
                        self.current_step = "LOCAL SENSEMAKING"
                        self.step_history.append(self.current_step)

                        response = self.invoke_with_retry(self.sense_making_agent, 'invoke_local_sense', {
                            'results': results,
                            'data_type': database,
                            'user_query': request
                        })
                        if (response != "FAILED"):
                            # check if response is NoneType
                            if response is None:
                                local_sense = "Failed to generate local sense of results because response is None."
                            elif ("NOT POSSIBLE" not in response):
                                local_sense = response["summary"]
                            else:
                                local_sense = f"Failed to generate local sense of results because {response['NOT POSSIBLE']}"
                        else:
                            local_sense = f"Failed to generate local sense of results because of LLM call failure"

                    question_and_sense = f"Question: \n {request} \n\n Database: \n {database} \n\n Answer: \n {local_sense}\n\n"
                    self.memory += "\n\n" + question_and_sense

                    print("Sense making step")
                    self.current_step = "GLOBAL SENSEMAKING"
                    self.step_history.append(self.current_step)

                    response = self.invoke_with_retry(self.sense_making_agent, 'invoke_global_sense', {
                        'user_query': self.user_query,
                        'understanding': self.understanding,
                        'memory': self.memory,
                        "hypothesis": self.hypothesis
                    })
                    if response is not None and response != "FAILED" and 'understanding' in response:
                        self.understanding = response['understanding']
                    num_iters += 1
                    print("Number of iterations: ", num_iters)

            if state == "END" or state =='END' or num_iters >= max_iters:
                state = "END"
                print("\n\n**********************************************************")
                print("End of sense making")
                print("**********************************************************")
                print(self.understanding)
                self.current_step = "PRESENTATION"
                self.step_history.append(self.current_step)
                answer = self.invoke_with_retry(self.presentation_agent, 'invoke', {
                    'user_query': self.user_query,
                    'understanding': self.understanding,
                    'instructions': self.presentation_instructions
                })
                self.answer = answer["response"]
                print("******************* ANSWER ***************************************")
                print(self.answer)
                self.current_step = "FINISH"
                self.step_history.append(self.current_step)
                break
            if state != "END" or state != 'INF':
                print("Invalid state")
                print(state)
                self.step_history.append("INVALID STATE")



    def invoke_with_retry(self, agent, method, params, max_retries=1):
        retries = 0
        while retries <= max_retries:
            # print(f"Invoking {method} on {agent.__class__.__name__} with params: {params}")
            try:
                return getattr(agent, method)(params)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    print(f"Failed after {max_retries + 1} attempts: {str(e)}")
                    return "FAILED"

if __name__ == "__main__":
    presentation_instructions_ = '''
    clear and concise answer to the user query
    '''
    SenseMaker(
        "on 19th feb 25, can you tell me the list of postures and activities pilot2 did from 9am-10am? Response with a list in the format of {start time}-{end time{}: {posture}, {activities}. List activities for the entire one hour period.",
        presentation_instructions_).make_sense()
    a = ""