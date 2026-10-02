import asyncio
import logging
from autogen_agentchat.agents import CodeExecutorAgent, CodingAssistantAgent
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import RoundRobinGroupChat, StopMessageTermination
from autogen_ext.code_executor.docker_executor import DockerCommandLineCodeExecutor
from autogen_ext.models import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
import os

async def main() -> None:
    from autogen_ext.models import AzureOpenAIChatCompletionClient
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    # Create the token provider
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAIChatCompletionClient(
        model="PROPILOT",
        api_version="2024-05-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        model_capabilities={
            "vision": True,
            "function_calling": True,
            "json_output": True,
        }
    )
    async with DockerCommandLineCodeExecutor(work_dir="coding", image="sensemaking-code", timeout=600) as code_executor:
        code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
        coding_assistant_agent = CodingAssistantAgent(
            "coding_assistant", model_client= client
        )
        group_chat = RoundRobinGroupChat([coding_assistant_agent, code_executor_agent])
        result = await group_chat.run(
            task="create dummy data and calculate its mean and standard deviation",
            termination_condition=StopMessageTermination(),
        )


asyncio.run(main())
