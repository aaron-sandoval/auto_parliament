# %%
import os
from dotenv import load_dotenv
from pathlib import Path

from inspect_ai.log import EvalLog

from ethics_eval import run_eval
from eval_datasets import ethics_datasets


models = ["openai/gpt-4o-mini"]

# %%
log: EvalLog = run_eval(ethics_datasets[0], models[0])

# %%
if log[0].status == "success":
    print("Eval completed successfully")
    log_path: str = str(Path("data/eval_logs"))
    !inspect view start --log-dir $log_path --port 7575
else:
    print("Eval failed")
# %%
# !inspect view start --log-dir $log_path --port 7575