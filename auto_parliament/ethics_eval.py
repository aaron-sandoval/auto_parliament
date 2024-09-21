from typing import Literal
from functools import cache

from datasets import load_dataset, DatasetDict, Dataset
from autogen import AssistantAgent, ChatResult
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from parliament import agents

