
from tensorpack import AugmentImageComponents

from .fileflow import get_fileflow
from ..common import parse_value
from .fileflow import get_fileflow
from .imgaug import augment, ImageAugmentorListProxy, NotSafeAugmentorList, CropFlow
from .common import ReadFilesFlow, PrintImageFlow

__all__ = ['get_train_data']


def get_crop_ds(data_cfg, crop_cfg):
    def wrap(ds, data_cfg=data_cfg, crop_cfg=crop_cfg):
        size = data_cfg.batch_shape
        crop_cfg = parse_value(crop_cfg)
        number =  crop_cfg.value
        assert type(data_cfg.inputs) == list
        scales = crop_cfg.scales if crop_cfg.scales is not None else [1] * len(data_cfg.inputs)
        return CropFlow(ds, size, number, scales=scales)
    return wrap

def get_print_ds(print_cfg):
    def wrap(ds, print_cfg=print_cfg):
        assert print_cfg.path is not None and print_cfg.each is not None, print_cfg
        return PrintImageFlow(ds, print_cfg.path, each = print_cfg.each, clear=print_cfg.clear)
    return wrap

def get_train_data(cfg, endless=True):
    ds_imgs = ReadFilesFlow(get_fileflow(cfg, endless=endless))

    if cfg.aug:
        assert type(cfg.aug) == list,type(cfg.aug)

        augs = []
        constructors = []
        for aug in cfg.aug:
            name = aug
            value = None
            if isinstance(aug, dict):
                assert len(aug) == 1, aug
                name = list(aug.keys())[0]
                value = aug[name]
            if name == 'resize':
                augs.append(ImageAugmentorListProxy(augment.Resize(value), shared_params=True))
            if name == 'crop':
                constructors.append(get_crop_ds(cfg, value))
            if name == 'print':
                constructors.append(get_print_ds(value))

        print("Agmentators", len(augs))
        ds_imgs = AugmentImageComponents(ds_imgs, NotSafeAugmentorList(augs), index=list(range(len(cfg.inputs))))

        for ds_create in constructors:
            ds_imgs = ds_create(ds_imgs)
    return ds_imgs
