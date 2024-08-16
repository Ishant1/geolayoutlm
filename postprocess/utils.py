import numpy as np
import torch

def create_bbox_token_list(input_ids, bbox_first_index, mask):
    input_ids = input_ids.numpy()
    bbox_first_index = bbox_first_index.numpy()
    mask = mask.numpy()

    valid_input_ids = input_ids[mask==1]
    valid_bbox_first_index = bbox_first_index[mask==1]

    all_list = []
    bbox_list = []

    for i in range(len(valid_input_ids)):
        if valid_bbox_first_index[i] or i==len(valid_input_ids)-1:
            if bbox_list!=[]:
                all_list.append(bbox_list)
            bbox_list = [valid_input_ids[i]]
        else:
            if bbox_list!=[]:
                bbox_list.append(valid_input_ids[i])
    return all_list


def get_largest_bbox(list_of_bbox):
    list_of_bbox = np.array(list_of_bbox)
    return [
        list_of_bbox[:, 0].min(),
        list_of_bbox[:, 1].min(),
        list_of_bbox[:, 2].max(),
        list_of_bbox[:, 3].max()
    ]


def agg_labels(labels):
    final_label = "O"
    unique_labels = np.unique(labels)
    for ul in unique_labels:
        if ul.startswith("B"):
            final_label = ul
    return final_label


def convert_tensors_to_device(input, device="cpu"):
    if isinstance(input, dict):
        return {i:convert_tensors_to_device(v, device) for i, v in input.items()}
    elif isinstance(input, list):
        return [convert_tensors_to_device(x, device) for x in input]
    elif isinstance(input, torch.Tensor):
        return input.to(device)

    return input
