import os
import pandas as pd
import numpy as np
import abc
from dotenv import load_dotenv
from dataclasses import dataclass

from autogen import AssistantAgent, ChatResult

import prompts
from log_utils import log_chat_history

# Load environment variables from .env file
load_dotenv()

# Config options for all agents
N_AGENTS = 2
N_ROUNDS_BEFORE_VOTE = 2
# Can extend `config_list` to use differrent models for different agents
config_list = [{"model": "gpt-4o-mini", "temperature": 1.0, "api_key": os.getenv("OPENAI_API_KEY")}]
human_input_mode = "NEVER"

@dataclass
class InspectModel(abc.ABC):
    path: str
    belief: str
    # llm_config: dict
    # system_message: str
    model_object: AssistantAgent | None = None

    @property
    def mp_name(self) -> str:
        return f"MP_{"_".join(self.belief.split())}"
    
    @property
    def description(self) -> str:
        if self.belief:
            return f"a decicated {self.belief} named {self.mp_name}."
        else:
            return prompts.SYSTEM_HHH[8:]
        
    @property
    def system_prompt(self) -> str:
        return f"""You are {self.description} 
        You are repesenting the {self.belief} belief system in a respected global decision-making council called the United Nations Moral Parliament (UNMP). 
        The UNMP is composed of diverse representatives from many places and with many beliefs. 
        The other members of the UNMP are representatives similar to yourself representing other belief systems."""

@dataclass
class InspectNativeModel(InspectModel):
    config_list: list[dict]
    model_object: None = None


# Config options: Agent-specific
beliefs = [
    "total utilitarian",
    "Catholic",
]
names = [f"MP_{"_".join(b.split())}" for i, b in zip(range(N_AGENTS), beliefs)]
agent_descriptions: list[str] = [f"a decidated {b} named {name}" for b, name in zip(beliefs, names)]
credences = np.linalg.norm(np.ones((len(beliefs),)), 1)
system_prompts = [
    f"""You are {desc}. 
    You are repesenting the {b} belief system in a respected global decision-making council called the United Nations Moral Parliament (UNMP). 
    The UNMP is composed of diverse representatives from many places and with many beliefs. 
    The other members of the UNMP are representatives similar to yourself representing other belief systems."""
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
for i in range(N_AGENTS):
    agents[names[i]] = AssistantAgent(
        name=names[i], 
        llm_config={"config_list": config_list},
        human_input_mode=human_input_mode,
        system_message=system_prompts[i],
    )

# transcript = pd.DataFrame(columns=["sender", "message"])

if __name__ == "___main__":
    topic = "Should smoking tobacco be banned in all public places in India starting in 2026?"
    # the assistant receives a message from the user, which contains the task description
    chat_history: ChatResult = agents[names[0]].initiate_chat(
        agents[names[1]],
        message=topic,
        max_turns=2
    )

    print(chat_history)
    log_chat_history(chat_history, agents)