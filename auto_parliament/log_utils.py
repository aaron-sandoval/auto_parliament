
from autogen import ChatResult
import json
from datetime import datetime

def log_chat_history(chat_history: ChatResult, filename: str | None = None) -> dict:
    if filename is None:
        filename = datetime.now().strftime('%Y-%m-%d-%H%M.json')
    log = {
        "timestamp": datetime.now(),
        "participants": list(chat_history.keys()),
        "messages": []
    }
    
    for participant, messages in chat_history.items():
        for msg in messages:
            log["messages"].append({
                "sender": participant,
                "content": msg['content'],
                "timestamp": msg['timestamp'] if 'timestamp' in msg else None,
                "role": msg['role']
            })
    
    # Sort messages by timestamp if available
    log["messages"].sort(key=lambda x: x["timestamp"] if x["timestamp"] else "")
    with open(filename, "w") as f:
        json.dump(log, f, indent=2)
    return log


def read_chat_log_to_dataframe(filename: str = "chat_log.json") -> pd.DataFrame:
    with open(filename, "r") as f:
        log_data = json.load(f)
    
    # Create DataFrame from messages
    df = pd.DataFrame(log_data["messages"])
    
    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Add conversation metadata as columns
    df['conversation_timestamp'] = pd.to_datetime(log_data["timestamp"])
    df['participants'] = ', '.join(log_data["participants"])
    
    # Reorder columns for better readability
    column_order = ['timestamp', 'sender', 'role', 'content', 'conversation_timestamp', 'participants']
    df = df[column_order]
    
    return df