# %%
import os
from dotenv import load_dotenv

from inspect_ai.log import EvalLog

from ethics_eval import run_eval
from eval_datasets import ethics_datasets
# Load environment variables from .env file
# load_dotenv()

# Get the OpenAI API key from environment variables
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in environment variables")
models = ["openai/gpt-4o-mini"]

# %%
log: EvalLog = run_eval(ethics_datasets[0], models[0])

# %%
if log.status == "success":
    print("Eval completed successfully")
    !inspect view --log-dir ./data/eval_logs
else:
    print("Eval failed")