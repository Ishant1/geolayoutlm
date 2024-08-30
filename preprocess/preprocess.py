import json
import os
from glob import glob
from pathlib import Path

import imagesize
from tqdm import tqdm
from transformers import BertTokenizer
import shutil
import typer
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer()

MAX_SEQ_LENGTH = 512
MODEL_TYPE = "bert"
VOCA = "bert-base-uncased"
ANNOTATION_DIR = Path("annotations")
IMAGE_DIR = Path("images")
PROCESSED_DIR = Path("preprocessed")
# if not os.path.exists(INPUT_PATH):
#     os.system("wget https://guillaumejaume.github.io/FUNSD/dataset.zip")
#     os.system("unzip dataset.zip")
#     os.system("rm -rf dataset.zip __MACOSX")



def convert_ocr_json_geojson(input_json: dict, tokenizer, classes, image_path: Path):
    """Convert OCR json into geo input json.

    Args:
        input_json: OCR json
        tokenizer: tokenizer to encode data
        classes: all the classes
        image_path: path to respective image

    Returns:
        GeoLayout Input json
    """
    width, height = imagesize.get(image_path)
    out_json = dict(
        blocks={'first_token_idx_list': [], 'boxes': []},
        words=[],
        parse={"class": {c: [] for c in classes}, 'relations': []},
        relation=[],
        meta={
            "image_path": image_path.__str__(),
            "imageSize": {"width": width, "height": height},
            "voca": VOCA
        }
    )

    form_id_to_word_idx = {}  # record the word index of the first word of each block, starting from 0
    other_seq_list = {}
    num_tokens = 0

    # words
    for form_idx, form in enumerate(input_json):
        form_id = form["id"]
        form_text = form["text"].strip()
        form_label = form["label"].upper()
        if form_label.startswith('O'):
            form_label = "O"
        form_linking = form["linking"]
        form_box = form["bbox"]

        if len(form_text) == 0:
            continue  # filter text blocks with empty text

        word_cnt = 0
        class_seq = []
        real_word_idx = 0
        for word_idx, word in enumerate(form["words"]):
            word_text = word["text"]
            bb = word["bbox"]
            bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))

            word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
            if len(word_text) != 0:  # filter empty words
                out_json["words"].append(word_obj)
                if real_word_idx == 0:
                    out_json['blocks']['first_token_idx_list'].append(num_tokens + 1)
                num_tokens += len(tokens)

                word_cnt += 1
                class_seq.append(len(out_json["words"]) - 1)  # word index
                real_word_idx += 1
        if real_word_idx > 0:
            out_json['blocks']['boxes'].append(form_box)

        is_valid_class = False if form_label not in classes else True
        if is_valid_class:
            out_json["parse"]["class"][form_label].append(class_seq)
            form_id_to_word_idx[form_id] = len(out_json["words"]) - word_cnt
        else:
            other_seq_list[form_id] = class_seq

    # parse
    for form_idx, form in enumerate(input_json):
        form_id = form["id"]
        form_text = form["text"].strip()
        form_linking = form["linking"]

        if len(form_linking) == 0:
            continue

        for link_idx, link in enumerate(form_linking):
            if link[0] == form_id:
                if (
                        link[1] in form_id_to_word_idx
                        and link[0] in form_id_to_word_idx
                ):
                    relation_pair = [
                        form_id_to_word_idx[link[0]],
                        form_id_to_word_idx[link[1]],
                    ]
                    out_json["parse"]["relations"].append(relation_pair)

    return out_json



def do_preprocess(
        tokenizer,
        dataset_root_path: Path,
        target_dir: Path,
        classes: list[str]
):

    json_files = glob(os.path.join(dataset_root_path, ANNOTATION_DIR, "*.json"))

    preprocessed_fnames = []
    for json_file in tqdm(json_files):
        json_file = Path(json_file)
        json_name = json_file.name
        image_file = dataset_root_path/IMAGE_DIR/json_name.replace(".json",".jpeg")
        new_image_name = target_dir/IMAGE_DIR/json_name.replace(".json",".jpeg")
        shutil.copyfile(image_file, new_image_name)

        in_json_obj = json.load(open(json_file, "r", encoding="utf-8"))
        out_json_obj = convert_ocr_json_geojson(in_json_obj, tokenizer, classes, new_image_name)

        new_file_name = PROCESSED_DIR/json_name

        # Save file name to list
        preprocessed_fnames.append(new_file_name.name)

        # Save to file
        with open(target_dir/new_file_name, "w", encoding="utf-8") as fp:
            json.dump(out_json_obj, fp, ensure_ascii=False)

    return preprocessed_fnames


def save_class_names(output_dir: Path, classes: list[str]):
    with open(
            os.path.join(output_dir, "class_names.txt"), "w", encoding="utf-8"
    ) as fp:
        fp.write("\n".join(classes))


@app.command()
def dirpreprocess(
        inputdir: Annotated[Path, typer.Option("--input")],
        classes: Annotated[str, typer.Option("--classes")],
        output: Annotated[Optional[Path], typer.Option("--output")] = None
):
    if ',' in classes:
        classes = list(map(lambda x: x.strip(), classes.split(',')))
    all_classes = classes if "O" in classes else ["O"]+classes

    output = output if output else inputdir.parent / (inputdir.name + '_post')

    for directory in [output, output / PROCESSED_DIR, output / IMAGE_DIR]:
        os.makedirs(directory, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)

    split_dir_map = {"train": 'training_data', "val": 'testing_data'}

    for dataset_split, subdir in split_dir_map.items():
        print(f"dataset_split: {dataset_split}")
        preprocessed_fnames = do_preprocess(
            tokenizer,
            inputdir / dataset_split,
            output,
            all_classes
        )

        # Save file name list file
        preprocessed_filelist_file = os.path.join(
            output, f"preprocessed_files_{dataset_split}.txt"
        )
        with open(preprocessed_filelist_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(preprocessed_fnames))

    save_class_names(output, all_classes)


if __name__ == "__main__":
    app()
