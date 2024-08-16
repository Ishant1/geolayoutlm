import json
import os
from pathlib import Path

from anls import anls_score
import re

from preprocess.floorplan.schemas import OcrFileOutput, FloorplanEntity, FloorplanDocument, FloorplanSplitDocument
import numpy as np


def name_validator(name):
    clean_name = re.sub("\d","",name).strip().lower()
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


def load_all_results(filepath, unique=True):
    with open(filepath, "r") as f:
        label_results = json.load(f)

    cleaned_labels = {}

    for key, label_values in label_results.items():
        if not unique:
            cleaned_labels[key] = FloorplanEntity.parse_obj(label_values)
        else:
            rooms = label_values['rooms']
            cleaned_rooms = []
            for r in rooms:
                # r['name'] = name_validator(r['name'])
                cleaned_rooms.append(r)

            unqiue_rooms = np.unique([x["name"] for x in cleaned_rooms])
            unique_rooms_dict = []
            for room_section in unqiue_rooms:
                filtered_rooms = list(filter(lambda x: x['name'] == room_section, cleaned_rooms))
                dimension_list = [x['dimension'] for x in filtered_rooms if x['dimension']]
                unique_rooms_dict.append({'name': room_section, 'dimension': dimension_list})
            label_values['rooms'] = unique_rooms_dict
            cleaned_labels[key] = FloorplanEntity.parse_obj(label_values)

    return cleaned_labels


def match_labels_and_linking(tess_dict, related_labels=None):
    tess_dict['label'] = None
    linking_list = [[]]*len(tess_dict['label'])

    if related_labels:
        rooms = sorted(related_labels.rooms, key=lambda x: len(x.name), reverse=True)
        for r in rooms:
            dimension_cleaned_col = tess_dict['text'].str.upper().str.replace(" ", "").str.replace("M", "")
            room_name = r.name.upper().replace(" ", "").replace("M", "")
            room_index = tess_dict.index[dimension_cleaned_col.str.contains(room_name)&(tess_dict['label'].isna())]
            room_index = int(room_index[0]) if len(room_index) else None
            if type(room_index)!=int:
                room_indexes = tess_dict.index[
                    dimension_cleaned_col.apply(lambda x: room_name.__contains__(x)) & (tess_dict['label'].isna()) & (tess_dict['text'].apply(lambda x: len(x)>3))
                ]
                if len(room_indexes)>1:
                    tess_dict.loc[room_indexes, 'label'] = 'ROOM_NAME'
                    room_index = min(room_indexes)
                    for idx in room_indexes:
                        if idx !=room_index:
                            linking_list[idx] = linking_list[idx]+[[room_index, idx]]
                            linking_list[room_index] = linking_list[room_index]+[[room_index, idx]]
            else:
                tess_dict.loc[room_index,'label'] = 'ROOM_NAME'
                # room_block = tess_dict.loc[room_index,'block_num']
                # room_line = tess_dict.loc[room_index,'line_num']
                if r.dimension:
                    dimension_cleaned = r.dimension.upper().replace(" ","").replace("M","")
                    dim_index = tess_dict.index[
                        dimension_cleaned_col.str.contains(dimension_cleaned)&(tess_dict['label'].isna())
                    ]
                    dim_index = int(dim_index[0]) if dim_index.any() else None
                    if not dim_index:
                        dim_index = tess_dict.index[dimension_cleaned_col.apply(
                            lambda x: anls_score(x, [dimension_cleaned]) >= 0.9
                        )]
                        dim_index = int(dim_index[0]) if dim_index.any() else None

                    if not dim_index:
                        dim_index = tess_dict.index[dimension_cleaned_col.apply(
                            lambda x: re.sub(r'\D','',x)==re.sub(r'\D','',dimension_cleaned)
                        )]
                        dim_index = int(dim_index[0]) if dim_index.any() else None
                    if type(dim_index)==int:
                        tess_dict.loc[dim_index, 'label'] = 'DIMENSION'
                        linking_list[room_index]= linking_list[room_index]+[[room_index, dim_index]]
                        linking_list[dim_index]= linking_list[dim_index]+[[room_index, dim_index]]

    tess_dict['linking'] = linking_list
    tess_dict['label'].fillna('others', inplace=True)
    tess_dict.reset_index(inplace=True)
    tess_dict.rename({'index': 'id'}, axis=1, inplace=True)

    return tess_dict.to_dict('records')


def create_floorplan_document(
        key: str,
        document_ocr: OcrFileOutput | None = None,
        entity: FloorplanEntity | None = None,
) -> FloorplanDocument:
    floorplan_document = FloorplanDocument(
        key=key,
        bbox= [ocr_output.bbox for ocr_output in document_ocr.ocr_result] if document_ocr else None,
        word= [ocr_output.text for ocr_output in document_ocr.ocr_result] if document_ocr else None,
        entity=entity
    )

    return floorplan_document


def create_split_from_document(
        document: FloorplanDocument
) -> FloorplanSplitDocument:

    floorplan_document_dict = document.dict()
    floorplan_document_dict.update({'index': 0})
    floorplan_split = FloorplanSplitDocument.parse_obj(
        floorplan_document_dict
    )
    return floorplan_split


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

        with open(json_dir / json_file) as f:
            json.dump(ocr_json, f)






