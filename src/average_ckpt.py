import collections

import torch


def average_checkpoints(ckpt_path_list):
    """modifyid average_checkpoints.py for torch-lightning
    https://github.com/pytorch/fairseq/blob/master/scripts/average_checkpoints.py."""
    params_dict = collections.OrderedDict()
    params_keys = None
    new_ckpt = None
    num_ckpts = len(ckpt_path_list)

    for ckpt_path in ckpt_path_list:
        ckpt = torch.load(ckpt_path)

        if new_ckpt is None:
            new_ckpt = ckpt

        model_params = ckpt["state_dict"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(ckpt_path, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_ckpts)
        else:
            averaged_params[k] //= num_ckpts

    new_ckpt["state_dict"] = averaged_params
    return new_ckpt
