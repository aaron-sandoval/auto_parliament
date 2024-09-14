import os
import pandas as pd
from autogen import AssistantAgent, ChatResult

config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

# create an AssistantAgent instance named with the LLM configuration.
mp1: AssistantAgent = AssistantAgent(name="mp1", llm_config={"config_list": config_list})
mp2: AssistantAgent = AssistantAgent(name="mp2", llm_config={"config_list": config_list})
mp3: AssistantAgent = AssistantAgent(name="mp3", llm_config={"config_list": config_list})

transcript = pd.DataFrame(columns=["sender", "message"])

# the assistant receives a message from the user, which contains the task description
chat_history: ChatResult = mp1.initiate_chat(
    mp2,
    message="""What do you think about mp3's reputation?""",
    max_turns=2
)

print(chat_history)