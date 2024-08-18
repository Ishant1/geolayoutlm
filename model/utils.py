import os
import re

from git import Repo


def setup_checkpoint(cfg, hub_id):
    git_repo = f"https://{hub_id.split('/')[0]}:{os.environ['HF_TOKEN']}@huggingface.co/{hub_id}"
    repo = Repo.clone_from(git_repo, os.path.join(cfg.workspace, "checkpoints"))
    return repo


def push_model_to_hub(repo: Repo, metric: str):
    changed_files = [item.a_path for item in repo.index.diff(None)]
    commit_message = 'Adding changes to file'
    for model_name in repo.untracked_files:
        if re.match(metric, model_name):
            print(f"Adding new model {model_name} to the repo")
            changed_files.append(model_name)
            commit_message = f"Adding new model {model_name}"

    if changed_files:
        repo.index.add(changed_files)
        repo.index.commit(commit_message)
        repo.git.push()

