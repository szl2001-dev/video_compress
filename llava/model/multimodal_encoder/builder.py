import os
from .umt_encoder import UMTVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if "umt-hd" in vision_tower:
        return UMTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, image_size=448, **kwargs)
    elif "umt" in vision_tower:
        return UMTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
