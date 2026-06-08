import torch


def _merge_module_state(module, incoming_state):
    merged_state = module.state_dict()
    for key, value in incoming_state.items():
        merged_state[key] = value
    module.load_state_dict(merged_state)


def _filtered_state_dict(module, key_filter=None):
    state = module.state_dict()
    if key_filter is None:
        return state
    return {key: value for key, value in state.items() if any(token in key for token in key_filter)}


def _read_checkpoint(checkpoint_or_path, map_location="cpu"):
    if isinstance(checkpoint_or_path, (str, bytes)):
        return torch.load(checkpoint_or_path, map_location=map_location)
    return checkpoint_or_path


def build_lucid_state(net_lucid, optimizer=None, unet_key_filter=None):
    state = {
        "vae_lora_target_modules": net_lucid.target_modules_vae,
        "rank_vae": net_lucid.lora_rank_vae,
        "state_dict_unet": _filtered_state_dict(net_lucid.unet, unet_key_filter),
        "state_dict_vae": _filtered_state_dict(net_lucid.vae, ("lora", "skip")),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    return state


def save_lucid_state(net_lucid, optimizer, output_path, unet_key_filter=None):
    torch.save(build_lucid_state(net_lucid, optimizer, unet_key_filter), output_path)


def load_lucid_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location)


def load_lucid_components(vae, unet, checkpoint_or_path, map_location="cpu"):
    state = _read_checkpoint(checkpoint_or_path, map_location)
    if "state_dict_vae" in state:
        _merge_module_state(vae, state["state_dict_vae"])
    _merge_module_state(unet, state["state_dict_unet"])
    return state


def load_lucid_state(net_lucid, optimizer, checkpoint_path, map_location="cpu"):
    state = load_lucid_components(net_lucid.vae, net_lucid.unet, checkpoint_path, map_location)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return net_lucid, optimizer


def load_flare_state(model, checkpoint_path, map_location="cpu"):
    checkpoint = _read_checkpoint(checkpoint_path, map_location)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(prepare_flare_state_dict(state_dict))
    return checkpoint


def build_disentanglement_checkpoint(model, optimizer, scheduler, epoch, config):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }


def save_disentanglement_checkpoint(output_path, model, optimizer, scheduler, epoch, config):
    checkpoint = build_disentanglement_checkpoint(model, optimizer, scheduler, epoch, config)
    torch.save(checkpoint, output_path)
    return checkpoint


def load_disentanglement_checkpoint(model, optimizer, scheduler, checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(prepare_flare_state_dict(checkpoint["model_state_dict"]))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def prepare_flare_state_dict(state_dict):
    key_aliases = (
        ('flare_attention.refl_detector.', 'flare_attention.background_detector.'),
        ('feature_disentangle.refl_branch.', 'feature_disentangle.background_branch.'),
        ('reflectance_head.', 'background_head.'),
    )

    prepared = {}
    for key, value in state_dict.items():
        prepared_key = key
        for source_prefix, target_prefix in key_aliases:
            if key.startswith(source_prefix):
                prepared_key = target_prefix + key[len(source_prefix):]
                break
        prepared[prepared_key] = value
    return prepared