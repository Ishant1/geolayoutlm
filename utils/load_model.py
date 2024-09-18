import torch

from model import get_model, GeoLayoutLMVIEModel


def load_model_weight(net, pretrained_model_file):
    print("Loading ckpt from:", pretrained_model_file)
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")
    if "state_dict" in pretrained_model_state_dict.keys():
        pretrained_model_state_dict = pretrained_model_state_dict["state_dict"]
    new_state_dict = {}
    valid_keys = net.state_dict().keys()
    invalid_keys = []
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        if new_k in valid_keys:
            new_state_dict[new_k] = v
        else:
            invalid_keys.append(new_k)
    print(f"These keys are invalid in the ckpt: [{','.join(invalid_keys)}]")
    net.load_state_dict(new_state_dict)


def get_model_and_load_weights(cfg, pretrained_model_file=None, cuda=True, eval=True):
    cfg.pretrained_model_file = pretrained_model_file or cfg.pretrained_model_file
    # net = get_model(cfg)
    #
    # load_model_weight(net, pretrained_model_file)
    net = GeoLayoutLMVIEModel.from_pretrained(cfg.pretrained_model_file, cfg=cfg)

    if cuda:
        net.to("cuda")

    if eval:
        net.eval()

    return net





