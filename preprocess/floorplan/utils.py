import json
import os

import numpy as np

from preprocess.floorplan.schemas import OcrFileOutput

from preprocess.floorplan.preprocess import match_labels_and_linking


def save_ocr_result(
        filename: str,
        ocr_results: dict[str,OcrFileOutput]
) -> None:
    ocr_result_nested_dict = {i:v.dict() for i,v in ocr_results.items()}

    with open(filename, "w+") as f:
        json.dump(ocr_result_nested_dict, f)


def combine_ocr_bbox(bboxes):
    all_bboxes = np.array(bboxes)
    return [all_bboxes[:,0].min(), all_bboxes[:,1].min(), all_bboxes[:,2].max(), all_bboxes[:,3].max()]


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
