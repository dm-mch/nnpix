from pprint import pprint

from tensorpack import AugmentImageComponents

from .fileflow import get_fileflow
from .imgaug import ImageAugmentorListProxy, NotSafeAugmentorList
from .common import ReadFilesFlow
from ..registry import AugmentRegistry, DataFlowRegistry


__all__ = ['get_train_data']



def get_train_data(cfg, endless=True):
    ds_imgs = ReadFilesFlow(get_fileflow(cfg, endless=endless))

    if cfg.aug:
        assert type(cfg.aug) == list,type(cfg.aug)
        augs_reg = AugmentRegistry() # storage of all possible augmentators
        flow_reg = DataFlowRegistry() # dataflows
        print("augs_reg", augs_reg._reg)
        print("flows_reg", flow_reg._reg)
        for aug in cfg.aug:
            name = aug
            value = None
            if isinstance(aug, dict):
                assert len(aug) == 1, aug
                name = list(aug.keys())[0]
                value = aug[name]
            if augs_reg[name] is not None:
                shared = True if not isinstance(value, dict) or value.shared is None else bool(value.shared)
                index = list(range(len(cfg.inputs))) # all dp indexes
                if isinstance(value, dict) and value.dp_index is not None:
                    index = value.dp_index if isinstance(value.dp_index, list) else [value.dp_index]
                aug = ImageAugmentorListProxy(augs_reg[name](value, cfg), shared_params=shared)
                # stack dataflow
                ds_imgs = AugmentImageComponents(ds_imgs, NotSafeAugmentorList([aug]), index=index)
            elif flow_reg[name] is not None:
                ds_imgs = flow_reg[name](ds_imgs, value, cfg)
            else:
                print("WARNING: unsuported augmetator: ", name, "skiped...")
    return ds_imgs
