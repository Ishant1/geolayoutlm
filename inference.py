import json
import os.path
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from bros import BrosTokenizer
from lightning_modules.data_modules.vie_dataset import VIEDataset
from postprocess.postprocess import process_eval_dataset
from preprocess.floorplan.huggingface import get_floorplan_images, get_floorplan_images_with_path
from preprocess.floorplan.ocr import OcrEngine
from preprocess.floorplan.preprocess import match_labels_and_linking
from preprocess.floorplan.utils import get_room_dm_pairs, Counter
from preprocess.preprocess import convert_ocr_json_geojson
from utils import get_config, get_eval_kwargs_geolayoutlm_vie
from utils.load_model import get_model_and_load_weights
from utils.util import write_json, load_json


def get_ocr_engine():
    return OcrEngine()


def get_input_from_image(
        ocr_engine: OcrEngine,
        image: Path | list[Path],
        classes: list[str],
        tokenizer
):

    image = image if type(image)==list else [image]
    ocr_df = {}
    for img in Counter(image, log_at=0.1):
        try:
            ocr_df[img] = ocr_engine.get_result_from_a_file(img, block=True)
        except:
            continue
    ocr_labels = {i: match_labels_and_linking(v) for i,v in ocr_df.items()}
    geojson_input = []
    for img, ocr in ocr_labels.items():
        geojson_input.append(convert_ocr_json_geojson(ocr, tokenizer, classes, img))

    return geojson_input

def get_dataset(json_list, cfg, tokenizer, classes):

    mode = 'val'

    dataset = VIEDataset(
        json_list,
        cfg.dataset,
        cfg.task,
        "geolayoutlm",
        cfg.model.head,
        cfg.dataset_root_path,
        tokenizer,
        class_names=classes
    )

    batch_size = min(cfg[mode].batch_size, len(json_list))

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg[mode].num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return data_loader


def get_ocr_input(
    image_paths: dict[str, str],
    classes: list[str],
    write: str| None = None
):
    json_obj = {}
    image_local_paths = get_floorplan_images(image_paths)
    ocr_engine = get_ocr_engine()
    tokenizer = BrosTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    json_lists = get_input_from_image(ocr_engine, image_local_paths, classes, tokenizer)
    for v in json_lists:
        image_key = Path(v["meta"]["image_path"]).stem
        v['meta']['url'] = image_paths[image_key]
        json_obj[image_key] = v

    if write:
        write_json(json_obj, write, True)

    return json_obj


def get_model_result(
        image_paths: dict[str, str],
        model_path: Path,
        classes: list[str],
        cuda=True,
        ocr_json: dict | None = None,
        write = str | None,
):
    json_lists = ocr_json or get_ocr_input(image_paths, classes)
    cfg = get_config()
    eval_kwargs = get_eval_kwargs_geolayoutlm_vie(classes=classes)
    net = get_model_and_load_weights(cfg, model_path, cuda)
    model_existing_outputs = load_json(write) if write and os.path.exists(write) else {}
    json_lists = {i:v for i,v in json_lists.items() if i not in model_existing_outputs}
    dict_result = {}
    if json_lists:
        json_lists = get_floorplan_images_with_path(json_lists)
        dataset = get_dataset(list(json_lists.values()), cfg, net.tokenizer, classes=classes)
        processed_dfs = process_eval_dataset(net, dataset, eval_kwargs)
        dict_result = {i: get_room_dm_pairs(df) for i,df in processed_dfs.items()}

    if write:
        dict_result = {i: v.json() for i,v in dict_result.items()}
        write_json(dict_result, write, True)

    return model_existing_outputs| dict_result
