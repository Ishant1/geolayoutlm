import json
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from bros import BrosTokenizer
from lightning_modules.data_modules.vie_dataset import VIEDataset
from postprocess.postprocess import process_eval_dataset
from preprocess.floorplan.ocr import OcrEngine
from preprocess.floorplan.preprocess import match_labels_and_linking
from preprocess.floorplan.utils import get_room_dm_pairs
from preprocess.preprocess import convert_ocr_json_geojson
from utils import get_config, get_eval_kwargs_geolayoutlm_vie
from utils.load_model import get_model_and_load_weights


def get_ocr_engine():
    return OcrEngine()


def get_input_from_image(
        ocr_engine: OcrEngine,
        image: Path | list[Path],
        classes: list[str],
        tokenizer
):

    image = image if type(image)==list else [image]
    ocr_df = []
    for img in tqdm(image):
        ocr_df.append(ocr_engine.get_result_from_a_file(img, block=True))
    ocr_labels = list(map(match_labels_and_linking, ocr_df))
    geojson_input = []
    for ocr, img in zip(ocr_labels, image):
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
    image_paths: Path | list[Path] | list[str],
    classes: list[str],
    write: str| None = None
):
    ocr_engine = get_ocr_engine()
    tokenizer = BrosTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    json_lists = get_input_from_image(ocr_engine, image_paths, classes, tokenizer)
    if write:
        with open(write, 'w+') as f:
            json.dump(json_lists, f)
    else:
        return json_lists


def get_model_result(
        image_paths: Path | list[Path] | list[str],
        model_path: Path,
        classes: list[str],
        cuda=True,
        ocr_json: str | None = None
):
    if not ocr_json:
        json_lists = get_ocr_input(image_paths, classes)
    else:
        with open(ocr_json, "r") as f:
            json_lists = json.load(f)

    cfg = get_config()
    eval_kwargs = get_eval_kwargs_geolayoutlm_vie(classes=classes)
    net = get_model_and_load_weights(cfg, model_path, cuda)
    dataset = get_dataset(json_lists, cfg, net.tokenizer, classes=classes)
    processed_dfs = process_eval_dataset(net, dataset, eval_kwargs)
    return {i: get_room_dm_pairs(df) for i,df in processed_dfs.items()}
