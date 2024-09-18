from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, login
import os

from tqdm import tqdm

from preprocess.floorplan.utils import get_image_from_url

login(os.environ['HF_TOKEN'])
hf = HfApi()

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


def load_hf_dataset(repo_id):
    houses = load_dataset(path=repo_id, name="houses", split='train')
    index = load_dataset(path=repo_id, name="index", split='train')
    return houses, index


def get_floorplan_images(house_df, image_dir = "images"):
    house_df = house_df.loc[~house_df["floorplan_url"].isnull(), :]
    floorplan_url = house_df.set_index("id")["floorplan_url"].to_dict()
    floorplan_url = {i:v for i,v in floorplan_url.items() if v}
    all_paths = []
    image_dir = Path(image_dir)
    image_dir.mkdir(exist_ok=True)
    for dict_item in tqdm(floorplan_url.items()):
        i,v = dict_item
        try:
            image_path = image_dir/f"{i}.jpeg"
            get_image_from_url(v, image_path)
            all_paths.append(image_path)
        except:
            continue
    return all_paths



