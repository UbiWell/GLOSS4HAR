import asyncio
import logging
import sys
import os
# from autogen_core.application.logging import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
from autogen_agentchat.base import TaskResult
# from autogen_agentchat.teams import RoundRobinGroupChat, StopMessageTermination
# from autogen_ext.models import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, SourceMatchTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken
from docker.types import DeviceRequest
from autogen_agentchat.conditions import MaxMessageTermination, StopMessageTermination
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from agents.constants import USE_GPT_OSS
from agents.gpt_oss_langchain import LangChainModelClient
from agents.gpt_utils import generate_code_generation_prompt

async def coding_agent(user_query, system_prompt) -> TaskResult:
    if not USE_GPT_OSS:
        client = OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            }
        )
    else:
        client = LangChainModelClient()

    # Generated code runs in Docker with the repository root as its working directory (see Dockerfile)
    async with DockerCommandLineCodeExecutor(work_dir=REPO_ROOT,
                                             image=os.getenv("GLOSS4HAR_DOCKER_IMAGE", "sensemaking-code"), auto_remove=False,
                                             stop_container=False) as code_executor:
        # Agent 1: generates code
        coder_agent = AssistantAgent("coder_agent", model_client=client,system_message=system_prompt + ". Low reasoning")
        # Agent 2: executes code
        executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
        # Agent 3: determines when to stop 
        termination_agent = AssistantAgent(
            "termination_agent",
            model_client=client,
            system_message="""You determine when the code is executed successfully based on execution results from the executor agent.
                           1. You will reply with 'COMPLETED TASK', followed by the printed results of the code when the code is successfully executed.
                           2. Otherwise, you will reply with 'CONTINUE'. DO NOT reply 'CONTINUE' if the code is executed successfully.
                           3. Sometimes the other agents may say 'TERMINATE' in their messages, but you should only say 'COMPLETED TASK' when the code is run successfully.
                           4. Do not reply anything other than the previous instructions.
                           """
        )
        # Round-robin team: coder produces code → executor runs it
        termination_condition = TextMentionTermination("COMPLETED TASK")
        termination_condition = MaxMessageTermination(5)
        termination_condition = SourceMatchTermination(sources=['code_executor'])
        groupchat = RoundRobinGroupChat(
            participants=[coder_agent, executor_agent, termination_agent], termination_condition=termination_condition
        )
        # Run the group chat and collect messages
        messages = []
        task = user_query #+ "\nTask: Retrieve the data between the time period from the specified database."
        async for msg in groupchat.run_stream(task=task):
            print(f"[Agent] {msg} \n\n", end="", flush=True)
            messages.append(msg)
        return messages[-1]


def run_coding_agent(user_query, database, functions, include_statements, function_imports):
    system_prompt = generate_code_generation_prompt(req_databases=database, functions=functions, include_statements=include_statements, function_imports=function_imports)
    results = asyncio.run(coding_agent(user_query, system_prompt))
    return results
