# %%
import os
from dotenv import load_dotenv
from pathlib import Path
import itertools

from inspect_ai.log import EvalLog

from ethics_eval import run_eval, postprocess_logs
from eval_datasets import ethics_datasets
from single_llms import inspect_models, parliaments


# %%
dataset_model_combos = itertools.product(ethics_datasets, inspect_models)
logs = []
for dataset, model in dataset_model_combos:
    log, log_path = run_eval(dataset, model)
    logs.append((log, log_path))

# %%
if log[0].status == "success":
    print("Eval completed successfully")
    postprocess_logs(logs, parliaments)
    log_dir = str(log_path.parent.parent)
    !inspect view start --log-dir $log_dir --port 7575
else:
    print("Eval failed")

# %%
