import torch
import pandas as pd
import json

from tqdm import tqdm

from lightning_modules.geolayoutlm_vie_module import parse_relations, parse_str_from_seq, parse_relations2
from model import GeoLayoutLMVIEModel
from postprocess.utils import get_largest_bbox, create_bbox_token_list, agg_labels, convert_tensors_to_device


def process_result_from_batch(result_head, data_object, eval_kwargs, model) -> list:
    processed_results = []
    for index in range(len(data_object["bio_labels"])):
        actuals = parse_str_from_seq(data_object["bio_labels"][index], data_object["are_box_first_tokens"][index],
                                     eval_kwargs["bio_class_names"])
        predictions = parse_str_from_seq(result_head["Logits4labeling"][index],
                                         data_object["are_box_first_tokens"][index], eval_kwargs["bio_class_names"])
        bboxes = list(map(get_largest_bbox,
                          create_bbox_token_list(data_object["bbox"][index], data_object["are_box_first_tokens"][index],
                                                 data_object["attention_mask"][index])))
        words = list(map(model.tokenizer.decode, create_bbox_token_list(data_object["input_ids"][index],
                                                                        data_object["are_box_first_tokens"][index],
                                                                        data_object["attention_mask"][index])))
        if result_head["max_prob_as_father"]:
            prob_linking = torch.sigmoid(result_head["logits4linking_list"][-1])[index]
            pr_el_labels = torch.where(
                prob_linking >= 0.5,
                prob_linking,
                torch.zeros_like(result_head["Logits4linking_list"][-1]))[index]
        else:
            pr_el_labels = result_head["pred4linking"][index]
        gt_relations, gt_s_memo, flag = parse_relations(
            data_object["el_labels_blk"][index],
            data_object["first_token_idxes"][index],
            data_object["el_label_blk_mask"][index],
            data_object["bio_labels"][index],
            eval_kwargs["bio_class_names"],
            max_prob_as_father=result_head["max_prob_as_father"],
            max_prob_as_father_upperbound=result_head["max_prob_as_father_upperbound"]
        )

        pr_relations, pr_s_memo, flag = parse_relations2(
            pr_el_labels,
            data_object["first_token_idxes"][index],
            data_object["el_label_blk_mask"][index],
            result_head["Logits4labeling"][index],
            eval_kwargs["bio_class_names"],
            max_prob_as_father=result_head["max_prob_as_father"],
            max_prob_as_father_upperbound=result_head["max_prob_as_father_upperbound"],
        )

        ocr_info = pd.DataFrame({"words": words, "bbox": bboxes, "prediction": predictions, "actual": actuals})
        ocr_info["bbox"] = ocr_info["bbox"].astype("str")
        grouped_ocr_results = ocr_info.groupby("bbox", sort=False).agg(
            {
                "words": lambda x: " ".join(x),
                "prediction": lambda x: agg_labels(x),
                "actual": lambda x: agg_labels(x),
            }
        )

        grouped_ocr_results.reset_index(inplace=True)
        grouped_ocr_results["links"] = grouped_ocr_results.reset_index().apply(
            lambda x: [r for r in pr_relations if x["index"] == r[0] and x["prediction"] == "B-ROOM_NAME"], axis=1
        )
        grouped_ocr_results["bbox"] = grouped_ocr_results["bbox"].apply(json.loads)
        processed_results.append(grouped_ocr_results)

    return processed_results


def process_eval_dataset(model: GeoLayoutLMVIEModel, eval_loader, eval_kwargs):
    device = next(model.parameters()).device
    all_processed_results = {}
    for data_obj in tqdm(eval_loader):
        with torch.no_grad():
            result, loss_dict = model(convert_tensors_to_device(data_obj, device))
        torch.cuda.empty_cache()

        result_head = convert_tensors_to_device(result, "cpu")
        data_obj = {i: v for i, v in data_obj.items() if i not in ["image", "el_labels_seq", "el_label_seq_mask"]}
        result_head["logits4labeling"] = torch.argmax(result_head["logits4labeling"], -1)
        df_result = process_result_from_batch(model, result_head, data_obj, eval_kwargs)
        ids = map(lambda x: x.split('/')[-1].split('.')[0], data_obj["image_path"])
        all_processed_results.update({i: v for i, v in zip(ids, df_result)})

    return all_processed_results
