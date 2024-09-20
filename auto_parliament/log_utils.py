from typing import Sequence
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from autogen import ChatResult, AssistantAgent
from copy import deepcopy

def log_chat_history(chat_history: ChatResult, agents: Sequence[AssistantAgent], filename: str | None = None, file_suffix: str | None = None) -> dict:
    if filename is None:
        filename = "".join([datetime.now().strftime('%Y-%m-%d-%H%M%S'), f"_{file_suffix}" if file_suffix is not None else "", ".json"])
    filename = Path(".")/"data"/"chat_logs"/filename
    log = {
        "timestamp": datetime.now().strftime('%Y-%m-%d-%H:%M'),
        "participants": [],
        "messages": []
    }
    
    for agent in agents.values():
        LLM_config = deepcopy(agent.llm_config["config_list"][0])
        LLM_config.pop("api_key")
        log["participants"].append({
            "name": agent.name,
            "system_prompt": agent.system_message,
            "LLM_config": LLM_config
        })

    for message in chat_history.chat_history:
        participant = message["name"]
        # for msg in message:
        log["messages"].append({
            "sender": participant,
            "content": message['content'],
            # "role": message['role']
        })
    
    with open(filename, "w+", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return log

# def log_eval()