import json
import os
import re
from typing import Union

import numpy as np
from pydantic import BaseModel, field_validator

from preprocess.floorplan.schemas import OcrFileOutput, combine_ocr_bbox


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


def get_dim(dim_ids, room_id, results_df, max_dis = 1000):
    final_dis = {}

    for d in dim_ids:
        dim_distance = get_bbox_dis(room_id.loc[room_id, 'bbox'], room_id.loc[d, 'bbox'])
        if dim_distance < max_dis:
            final_dis = {
                "bbox": results_df.loc[d, "bbox"], "name": results_df.loc[d, "words"]
            }
            max_dis = dim_distance

    return final_dis | {"area": clean_dim(final_dis.get('name',''))}


def clean_dim(dim_string):
    dim_string = re.sub(r"[^x0-9\.]","", dim_string)
    try:
        cleaned_dims = dim_string.replace(" ",'').split('x') if 'x' in dim_string else dim_string.replace(". ",'').split(' ')
        area = np.round(np.multiply(*list(map(float,cleaned_dims))), 2)
        return area
    except:
        return None


class RoomInfo(BaseModel):
    name: str
    dimension: Union[str, None] = None
    area: Union[float, None] = None
    @field_validator("name", mode="before")
    def name_validator(cls, v):
        clean_name = re.sub("\d","",v).strip().lower()
        if re.findall("bed|suite", clean_name):
            return "bedroom"
        elif re.findall("reception|lounge|living", clean_name):
            if re.findall("kitchen|diner", clean_name):
                return "open plan living room"
            else:
                return "living room"
        elif re.findall("kitchen|diner|dining", clean_name):
            return "kitchen"

        else:
            return clean_name


class HouseRooms(BaseModel):
    rooms: list[RoomInfo]

    def get_bedrooms(self):
        return list(filter(lambda x: x.name=="bedroom", self.rooms))


def get_room(link_ids, results_df):
    room_bbox = combine_ocr_bbox(results_df.loc[link_ids, "bbox"].to_list())
    room_name = " ".join(results_df.loc[link_ids, "words"].to_list())
    return {"bbox": room_bbox, "name": room_name}

def get_bbox_dis(bbox1, bbox2):
    mean1 = np.array([np.mean(bbox1[0], bbox1[2]), np.mean(bbox1[1], bbox1[3])])
    mean2 = np.array([np.mean(bbox2[0], bbox2[2]), np.mean(bbox2[1], bbox2[3])])
    return np.abs(np.linalg.norm(mean1-mean2))


def get_room_dm_pairs(results_df):
    pairs = []
    for i, r in results_df.iterrows():
        if r['links']:
            room_ids = [i]
            dim_ids = []
            for l in r['links']:
                if l[1] in results_df.index:
                    if results_df.loc[l[1], "prediction"] == "B-ROOM_NAME":
                        room_ids.append(l[1])
                    elif results_df.loc[l[1], "prediction"] == "B-DIMENSION":
                        dim_ids.append(l[1])
                    else:
                        continue

            pairs.append(
                RoomInfo(
                    name = get_room(room_ids, results_df).get('name'),
                    dimension = get_dim(dim_ids, i, results_df).get('name', None),
                    area = get_dim(dim_ids, i, results_df).get('area', None)
                )
            )

    return HouseRooms(rooms = pairs)



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
        if not json_file.endswith(".json"):
            continue
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
