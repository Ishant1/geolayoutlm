import json
import os

from preprocess.floorplan.schemas import OcrFileOutput
from pathlib import Path


def save_ocr_result(
        filename: str,
        ocr_results: dict[str,OcrFileOutput]
) -> None:
    ocr_result_nested_dict = {i:v.dict() for i,v in ocr_results.items()}

    with open(filename, "w+") as f:
        json.dump(ocr_result_nested_dict, f)


def normalise_bbox(bbox, height, width):
    return [int(bbox[0]*1000/width),int(bbox[1]*1000/height),int(bbox[2]*1000/width),int(bbox[3]*1000/height)]


def bbox_within(bbox_large, bbox_small,buffer=2):
    return bbox_large[0] <= bbox_small[0]+buffer and bbox_large[1]<=bbox_small[1]+buffer and bbox_large[2]>=bbox_small[2]-buffer and bbox_large[3]>=bbox_small[3]-buffer


def join_block_and_words(block_ocr, word_ocr):
    final_results = []
    for block in block_ocr:
        block_words = list(map(lambda x: x.__dict__ ,filter(lambda x: bbox_within(block.bbox, x.bbox), word_ocr)))
        final_results.append(block.__dict__ | {'words':block_words})
    return final_results


def load_ocr_from_file(filename: str | None) -> dict[str,OcrFileOutput]:
    ocr_results = {}
    if filename and os.path.exists(filename):
        with open(filename, "r") as f:
            ocr_results = json.load(f)

    return {i:OcrFileOutput.parse_obj(v) for i,v in ocr_results.items()}


def add_imagedir_to_json(dataset_dir):
    dataset_dir = Path(dataset_dir)
    json_dir = dataset_dir/"preprocessed"
    for json_file in os.listdir(json_dir):
        with open(json_dir/json_file) as f:
            ocr_json = json.load(f)
        image_path = ocr_json["meta"]["image_path"]

        if os.path.exists(image_path):
            pass
        elif os.path.exists(dataset_dir/image_path):
            ocr_json["meta"]["image_path"] = (dataset_dir/image_path).as_posix()
        elif os.path.exists(dataset_dir/"images"/image_path.name):
            ocr_json["meta"]["image_path"] = (dataset_dir/"images"/image_path.name).as_posix()
        else:
            ValueError("Incorrect image root")

        with open(json_dir / json_file, "w") as f:
            json.dump(ocr_json, f)
