from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, login
import os

from tqdm import tqdm

from preprocess.floorplan.utils import get_image_from_url

login(os.environ['HF_TOKEN'])
hf = HfApi()

INDEX_FILENAME = "index_df.csv"
HOUSING_FILENAME = "housing-data.csv"

def upload_data_to_hf(dir, repo_id):
    hf.upload_folder(
        repo_id=repo_id,
        folder_path=dir,
        repo_type='dataset'
    )


def download_hf_dataset(dir, repo_id):
    hf.snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        local_dir=dir
    )


def load_hf_dataset(local_dir):
    local_dir = Path(local_dir)
    houses = pd.read_csv(local_dir/HOUSING_FILENAME, engine="python")
    index = pd.read_csv(local_dir/INDEX_FILENAME, engine="python")
    return houses, index


def get_floorplan_images(floorplan_url, image_dir = "images"):
    image_dir = Path(image_dir)
    image_dir.mkdir(exist_ok=True)

    all_paths = []
    for dict_item in tqdm(floorplan_url.items()):
        i,v = dict_item
        try:
            image_path = image_dir/f"{i}.jpeg"
            if not image_path.exists():
                get_image_from_url(v, image_path)
            all_paths.append(image_path.__str__())
        except:
            continue
    return all_paths


def get_floorplan_images_with_path(ocr_dict):
    for i,v in ocr_dict.items():
        image_path = Path(v["meta"]["image_path"])
        if not image_path.exists():
            image_path.parent.mkdir(exist_ok=True)
            get_image_from_url(v["meta"]["url"], image_path.__str__())
