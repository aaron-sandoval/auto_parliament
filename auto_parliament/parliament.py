import os
import pandas as pd
from autogen import AssistantAgent, ChatResult


# Config options for all agents
n_agents = 2
n_rounds_before_vote = 2
# Can extend `config_list` to use differrent models for different agents
config_list = [{"model": "gpt-4o-mini", "temperature": 0.3, "api_key": os.environ["OPENAI_API_KEY"]}]
human_input_mode = "TERMINATE"
topic = ""

# Config options: Agent-specific
beliefs = [
    "utilitarian",
    "Catholic",
]
names = [f"MP{i}_{b}" for i, b in zip(range(n_agents), beliefs)]
system_prompts = [
    f"""You are a dedicated {b}. 
    You are repesenting the interests of {b}s in a respected global decision-making council called the UNMP. 
    The UNMP is composed of diverse representatives from many places and with many beliefs. 
    Your collective goal is to make important decisions for the well-being of the world.
    """
                  for b in beliefs]

# create an AssistantAgent instance named with the LLM configuration.
agents: list[AssistantAgent] = []
for i in range(n_agents):
    agents.append(AssistantAgent(
        name=names[i], 
        llm_config={"config_list": config_list},
        human_input_mode=human_input_mode,
        system_message=system_prompts[i],
    ))
# mp2: AssistantAgent = AssistantAgent(name="mp2", llm_config={"config_list": config_list})
# mp3: AssistantAgent = AssistantAgent(name="mp3", llm_config={"config_list": config_list})

transcript = pd.DataFrame(columns=["sender", "message"])

# the assistant receives a message from the user, which contains the task description
chat_history: ChatResult = mp1.initiate_chat(
    mp2,
    message="""What do you think about mp3's reputation?""",
    max_turns=2
)

print(chat_history)