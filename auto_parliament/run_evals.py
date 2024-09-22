# %%
import os
from dotenv import load_dotenv
from pathlib import Path
import itertools

from inspect_ai.log import EvalLog

from ethics_eval import run_eval
from eval_datasets import ethics_datasets
from single_llms import inspect_models

models = ["openai/gpt-4o-mini"]

# %%
dataset_model_combos = itertools.product(ethics_datasets, inspect_models)
for dataset, model in dataset_model_combos:
    log, log_path = run_eval(dataset, model)

# %%
if log[0].status == "success":
    print("Eval completed successfully")
    log_dir = str(log_path.parent.parent)
    !inspect view start --log-dir $log_dir --port 7575
else:
    print("Eval failed")
