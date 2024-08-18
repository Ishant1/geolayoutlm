import os
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from lightning_modules.data_modules.vie_data_module import VIEDataModule
from lightning_modules.geolayoutlm_vie_module import GeoLayoutLMVIEModule
from model.utils import get_pre_trained_model
from preprocess.floorplan.utils import add_imagedir_to_json
from utils import get_callbacks, get_config, get_loggers, get_plugins

app = typer.Typer()

DATASET_SUB_DIR = "post"


def get_huggingface_data(
        dataset_name: Annotated[str, typer.Option("--input")] = "Aggish/goefloorplan",
        target_dir: Annotated[str, typer.Option("--input")] = "./GeoLayout",
):

    target_dir = Path(target_dir)
    subprocess.run([
        "huggingface-cli",
        "download",
        "--resume-download",
        "--repo-type",
        "dataset",
        dataset_name,
        "--local-dir",
        target_dir.parent.as_posix()
    ]
    )

    add_imagedir_to_json(os.path.join(target_dir,DATASET_SUB_DIR))



def finetune():


    cfg = get_config()
    # cfg["workspace"] = workspace if workspace else cfg["workspace"]
    # cfg["train"]["accelerator"] = device if device else cfg["train"]["accelerator"]
    # cfg["dataset_root_path"] = os.path.join(data_dir, DATASET_SUB_DIR) if data_dir else cfg["dataset_root_path"]
    # cfg["train"]["batch_size"] = batch_size if batch_size else cfg["train"]["batch_size"]
    # cfg["train"]["max_epochs"] = epochs if epochs else cfg["train"]["max_epochs"]
    print(cfg)

    if not os.path.exists(cfg['dataset_root_path']):
        get_huggingface_data(target_dir=cfg['dataset_root_path'])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
    seed_everything(cfg.seed)
    get_pre_trained_model(cfg)

    if cfg.model.head in ["vie"]:
        pl_module = GeoLayoutLMVIEModule(cfg)
    else:
        raise ValueError(f"Not supported head {cfg.model.head}")

    callbacks = get_callbacks(cfg)
    plugins = get_plugins(cfg)
    loggers = get_loggers(cfg)

    trainer = Trainer(
        accelerator=cfg.train.accelerator,
        gpus=torch.cuda.device_count(),
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
        callbacks=callbacks,
        plugins=plugins,
        sync_batchnorm=True,
        precision=16 if cfg.train.use_fp16 else 32,
        detect_anomaly=False,
        replace_sampler_ddp=False,
        move_metrics_to_cpu=False,
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=cfg.train.val_interval,
        logger=loggers,
        benchmark=cfg.cudnn_benchmark,
        deterministic=cfg.cudnn_deterministic,
        limit_val_batches=cfg.val.limit_val_batches,
    )

    # import ipdb;ipdb.set_trace()
    data_module = VIEDataModule(cfg, pl_module.net.tokenizer)

    trainer.fit(pl_module, datamodule=data_module)


if __name__ == "__main__":
    finetune()
