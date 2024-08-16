"""
Example:
    python evaluate.py --config=configs/finetune_funsd.yaml
"""

import os
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from glob import glob
import cv2

from lightning_modules.data_modules.vie_dataset import VIEDataset
from utils import get_config, get_eval_kwargs_geolayoutlm_vie
from utils.load_model import get_model_and_load_weights


def main():
    mode = "val"
    cfg = get_config()
    if cfg[mode].dump_dir is not None:
        cfg[mode].dump_dir = os.path.join(cfg[mode].dump_dir, cfg.workspace.strip('/').split('/')[-1])
    else:
        cfg[mode].dump_dir = ''
    print(cfg)

    if cfg.pretrained_model_file is None:
        pt_list = os.listdir(os.path.join(cfg.workspace, "checkpoints"))
        if len(pt_list) == 0:
            print("Checkpoint file is NOT FOUND!")
            exit(-1)
        pt_to_be_loaded = pt_list[0]
        if len(pt_list) > 1:
            # import ipdb;ipdb.set_trace()
            for pt in pt_list:
                if cfg[mode].pretrained_best_type in pt:
                    pt_to_be_loaded = pt
                    break
        cfg.pretrained_model_file = os.path.join(cfg.workspace, "checkpoints", pt_to_be_loaded)

    net = get_model_and_load_weights(cfg)


    if cfg.model.backbone in [
        "alibaba-damo/geolayoutlm-base-uncased",
        "alibaba-damo/geolayoutlm-large-uncased",
    ]:
        backbone_type = "geolayoutlm"
    else:
        raise ValueError(
            f"Not supported model: cfg.model.backbone={cfg.model.backbone}"
        )

    dataset = VIEDataset(
        f"preprocessed_files_{mode}.txt",
        cfg.dataset,
        cfg.task,
        backbone_type,
        cfg.model.head,
        cfg.dataset_root_path,
        net.tokenizer,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg[mode].batch_size,
        shuffle=False,
        num_workers=cfg[mode].num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if cfg.model.head == "vie":
        from lightning_modules.geolayoutlm_vie_module import (
            do_eval_epoch_end,
            do_eval_step
        )
        eval_kwargs = get_eval_kwargs_geolayoutlm_vie(cfg.dataset_root_path)
    else:
        raise ValueError(f"Unknown cfg.config={cfg.config}")

    step_outputs = []
    for example_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Convert batch tensors to given device
        device = next(net.parameters()).device
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            head_outputs, loss_dict = net(batch)
        step_out = do_eval_step(batch, head_outputs, loss_dict, eval_kwargs, dump_dir=cfg[mode].dump_dir)
        step_outputs.append(step_out)

    # Get scores
    scores = do_eval_epoch_end(step_outputs)
    if cfg.task != 'analysis':
        for task_name, score_task in scores.items():
            print(
                f"{task_name} --> precision: {score_task['precision']:.4f}, recall: {score_task['recall']:.4f}, f1: {score_task['f1']:.4f}"
            )
    else:
        print('eval: | ', end='')
        for key, value in scores.items():
            print(f"{key}: {value:.4f}", end=' | ')
        print()
    # Visualize
    if len(cfg[mode].dump_dir) > 0:
        visualize_tagging(cfg[mode].dump_dir)
        visualize_linking(cfg[mode].dump_dir)


def visualize_tagging(detail_path):
    pass


def visualize_linking(detail_path):
    file_paths = glob(os.path.join(detail_path, "*_linking.txt"))
    vis_dir = os.path.join(detail_path, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    for fp in tqdm(file_paths):
        with open(fp, 'r') as f:
            img_path = f.readline().strip('\n')
            img = cv2.imread(img_path, 1)
            f.readline()
            # read coord
            blk_coord_dict = {}
            while True:
                line = f.readline()
                if line.strip('\n') == '':
                    break
                blk_id, blk_coord = line.strip('\n').split('\t')
                blk_coord = [int(v) for v in blk_coord.split(',')]
                blk_coord_dict[blk_id] = blk_coord
            # read links and draw
            color_box = (205, 116, 24)
            color_lk = {"RIGHT": (0, 255, 0), "MISS": (59, 150, 241), "ERROR": (0, 0, 255)} # green, yellow, red
            while True:
                line = f.readline()
                if not line or line.strip('\n') == '':
                    break
                link, flag = line.strip('\n').split('\t')
                fthr_id, son_id = link.split(',')
                box_father = blk_coord_dict[fthr_id]
                cv2.rectangle(img, tuple(box_father[:2]), tuple(box_father[2:]), color_box, 2)
                center_father = ((box_father[0] + box_father[2]) // 2, (box_father[1] + box_father[3]) // 2)
                box_son = blk_coord_dict[son_id]
                cv2.rectangle(img, tuple(box_son[:2]), tuple(box_son[2:]), color_box, 2)
                center_son = ((box_son[0] + box_son[2]) // 2, (box_son[1] + box_son[3]) // 2)
                # link
                cv2.arrowedLine(img, center_father, center_son, color_lk[flag], thickness=2, tipLength=0.06)
        vis_fn = os.path.splitext(os.path.basename(fp))[0] + '.png'
        cv2.imwrite(os.path.join(vis_dir, vis_fn), img)


if __name__ == "__main__":
    main()
