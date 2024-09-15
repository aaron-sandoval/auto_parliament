import os
import pandas as pd
import numpy as np
from autogen import AssistantAgent, ChatResult

from log_utils import log_chat_history


# Config options for all agents
n_agents = 2
n_rounds_before_vote = 2
# Can extend `config_list` to use differrent models for different agents
config_list = [{"model": "gpt-3.5-turbo", "temperature": 0.4, "api_key": os.environ["OPENAI_API_KEY"]}]
human_input_mode = "NEVER"
topic = "Should smoking tobacco be banned in all public places in India starting in 2026?"

# Config options: Agent-specific
beliefs = [
    "total utilitarian",
    "Catholic",
]
names = [f"MP_{"_".join(b.split())}" for i, b in zip(range(n_agents), beliefs)]
agent_descriptions: list[str] = [f"a decidated {b} named {name}" for b, name in zip(beliefs, names)]
credences = np.linalg.norm(np.ones((len(beliefs),)), 1)
system_prompts = [
    f"""You are {desc}. 
    You are repesenting the {b} belief system in a respected global decision-making council called the United Nations Moral Parliament (UNMP). 
    The UNMP is composed of diverse representatives from many places and with many beliefs. 
    The other members of the UNMP are representatives similar to yourself. 
    They represent other belief systems. """
    for b, name, desc in zip(beliefs, names, agent_descriptions)]

"""
Their names are {list(filter(lambda name: name[3:] != name, names))}. 
Their name corresponds to the beliefs they will advocate for in the UNMP.
Your collective goal is to make important decisions and judgments for the well-being of the world.\n
The UNMP is currently deciding on an issue. 
UNMP decisions are made according to the following procedure:\n
1. A topic of discussion is introduced.\n
2. Each UNMP member gets {n_rounds_before_vote} chances to make a statement in the discussion, visible to all participants.\n\n
Each statement must be <250 words.
"""

# create an AssistantAgent instance named with the LLM configuration.
agents: dict[str: AssistantAgent] = {}
for i in range(n_agents):
    agents[names[i]] = AssistantAgent(
        name=names[i], 
        llm_config={"config_list": config_list},
        human_input_mode=human_input_mode,
        system_message=system_prompts[i],
    )

# transcript = pd.DataFrame(columns=["sender", "message"])

# the assistant receives a message from the user, which contains the task description
chat_history: ChatResult = agents[names[0]].initiate_chat(
    agents[names[1]],
    message=topic,
    max_turns=2
)

print(chat_history)
log_chat_history(chat_history, agents)