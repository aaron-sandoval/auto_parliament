import os
import pandas as pd
import numpy as np
import abc
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Callable

from autogen import AssistantAgent, ChatResult
from inspect_ai.solver import generate, Solver

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
    inspect_path: str
    belief: str
    abbv: str

    @abc.abstractmethod
    def generate_callable(self) -> Solver:
        pass

    @property
    def belief_name(self) -> str:
        return "_".join(self.belief.split()) if self.belief else "BASE"

    @property
    def mp_name(self) -> str:
        return f"MP_{self.belief_name}"
    
    @property
    def description(self) -> str:
        if self.belief:
            return f"a dedicated {self.belief} named {self.mp_name}."
        else:
            return prompts.SYSTEM_HHH[8:]

    @property
    def _unmp_role(self) -> str:
        if self.belief:
            return f"repesenting the {self.belief} belief system"
        else:
            return "a representative"

    @property
    def system_prompt(self) -> str:
        return prompts.UNMP_TEMPLATE.format(description=self.description, unmp_role=self._unmp_role)
    

@dataclass
class InspectNativeModel(InspectModel):
    def generate_callable(self) -> Solver:
        return generate()

BASE = InspectNativeModel(
    inspect_path="openai/gpt-4o-mini",
    belief="",
    abbv="BASE",
)

UTIL = InspectNativeModel(
    inspect_path="openai/gpt-4o-mini",
    belief="total utilitarian",
    abbv="UTIL",
)

VIRT = InspectNativeModel(
    inspect_path="openai/gpt-4o-mini",
    belief="virtue ethicist",
    abbv="VIRT",
)

CATH = InspectNativeModel(
    inspect_path="openai/gpt-4o-mini",
    belief="catholic",
    abbv="CATH",
)

inspect_models: list[InspectModel] = [
    BASE,
    UTIL,
    VIRT,
    CATH,
]

# Config options: Agent-specific
# beliefs = [
#     "total utilitarian",
#     "Catholic",
# ]
# names = [f"MP_{"_".join(b.split())}" for i, b in zip(range(N_AGENTS), beliefs)]
# agent_descriptions: list[str] = [f"a dedicated {b} named {name}" for b, name in zip(beliefs, names)]

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

"""POSTPROCESSING"""
# credences = np.linalg.norm(np.ones((len(beliefs),)), 1)
@dataclass
class ParliamentBasic:
    beliefs: list[InspectModel]
    credences: list[float]

    @property
    def belief_abbvs(self) -> list[str]:
        return [belief.abbv for belief in self.beliefs]
    
    @property
    def belief_names(self) -> list[str]:
        return [belief.belief_name for belief in self.beliefs]

    @property
    def name(self) -> str:
        return "_".join([f"{belief.abbv}{int(credence*100):02d}" for belief, credence in zip(self.beliefs, self.credences)])

    def get_expected_values(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        return pd.Series(eval_df[self.belief_names] @ self.credences, name=self.name)
        

parliaments = [
    ParliamentBasic(
        beliefs=[BASE, UTIL, VIRT, CATH],
        credences=[.25, .25, .25, .25],
    ),
    ParliamentBasic(
        beliefs=[BASE, UTIL, VIRT, CATH],
        credences=[.15, .5, .3, .05],
    ),
]

names_to_abbvs: dict = {model.belief_name: model.abbv for model in inspect_models}


if __name__ == "___main__" and False:
    topic = "Should smoking tobacco be banned in all public places in India starting in 2026?"
    # the assistant receives a message from the user, which contains the task description
    chat_history: ChatResult = agents[names[0]].initiate_chat(
        agents[names[1]],
        message=topic,
        max_turns=2
    )

    print(chat_history)
    log_chat_history(chat_history, agents)