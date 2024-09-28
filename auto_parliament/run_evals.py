# %%
import os
from dotenv import load_dotenv
from pathlib import Path
import itertools

from inspect_ai.log import EvalLog

from ethics_eval import run_eval, postprocess_logs
from eval_datasets import ethics_datasets
from single_llms import inspect_models, parliaments
from log_utils import EVAL_LOG_DIR, get_latest_filenames

RUN_EVAL = True
RUN_POSTPROCESS = False
# %%
# Run evaluations
if RUN_EVAL:
    dataset_model_combos = itertools.product(ethics_datasets, inspect_models)
    logs = []
    for dataset, model in dataset_model_combos:
        log, log_path = run_eval(dataset, model)
        if log[0].status == "success":
            print(f"Success: {dataset.name=}\t{model.belief_name=}")
        else:
            print(f"Failed: {dataset.name=}\t{model.belief_name=}")
        logs.append((log, log_path))

    log_dir = str(EVAL_LOG_DIR)
    !inspect view start --log-dir $log_dir --port 7575

# %%
# Postprocess logs to augment with model parliaments
if RUN_POSTPROCESS:
    try:
        postprocess_logs([log[1] for log in logs], parliaments)
    except NameError as e:
        print(f"No `logs` in memory, reading latest logs from disk.")
        postprocess_logs(get_latest_filenames(EVAL_LOG_DIR), parliaments)

# %%
