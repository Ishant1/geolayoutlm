import os
import re
from pathlib import Path
import urllib.request

from git import Repo


def setup_checkpoint(cfg, hub_id):
    git_repo = f"https://{hub_id.split('/')[0]}:{os.environ['HF_TOKEN']}@huggingface.co/{hub_id}"
    checkpoint_dir = os.path.join(cfg.workspace, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        repo = Repo.clone_from(git_repo, checkpoint_dir)
    else:
        repo = Repo(checkpoint_dir)

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


def get_pre_trained_model(cfg):
    if not os.path.exists(cfg['model']['model_ckpt']):
        parents = Path(cfg['model']['model_ckpt']).parent
        parents.mkdir(parents=True)
        urllib.request.urlretrieve(cfg['model']['base_url'], cfg['model']['model_ckpt'])