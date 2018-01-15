import numpy as np
from pprint import pprint

from tensorpack import AugmentImageComponents, RandomMixData, PrefetchDataZMQ, LocallyShuffleData

from .fileflow import get_fileflow
from .imgaug import ImageAugmentorListProxy, NotSafeAugmentorList, RemoveCustomParamsFlow
from .common import ReadFilesFlow, EndlessData
from ..registry import AugmentRegistry, DataFlowRegistry


__all__ = ['get_train_data']


def get_train_data(cfg, common_cfg, endless=True):
    assert cfg.inputs is not None and type(cfg.inputs) == list, cfg

    if np.alltrue([isinstance(s, str) for s in cfg.inputs]):
        # all inputs are paths: read files from there
        file_flow = get_fileflow(cfg, common_cfg, endless=endless)
        #print("Files input size", file_flow.size())
        input_flow = ReadFilesFlow(file_flow)
        #print("Readed input size", input_flow.size())
        input_flow = build_flow(input_flow, cfg, common_cfg, endless=endless)
        #print("Augmented input size", input_flow.size())
    elif np.alltrue([isinstance(d, dict) for d in cfg.inputs]):
        # all inputs are data
        dss = [get_train_data(inpt_cfg, common_cfg, endless=endless) for inpt_cfg in cfg.inputs]
        #print("DS for mix:", dss)
        print("Sizes:", [ds.size() for ds in dss])
        input_flow = RandomMixData(dss)
        print("Mixed size:", input_flow.size())
        if endless:
            input_flow = EndlessData(input_flow)
        input_flow = build_flow(input_flow, cfg, common_cfg, endless=endless)
    else:
        raise Exception("Not supported types of inputs, should be all paths or all dataflow cfg dictionary" + str(cfg))
    return input_flow

def build_flow(input_flow, cfg, common_cfg, endless=True):
    cfg = cfg.join(common_cfg)
    ds_imgs = input_flow

    if cfg.aug:
        assert type(cfg.aug) == list,type(cfg.aug)
        augs_reg = AugmentRegistry() # storage of all possible augmentators
        flow_reg = DataFlowRegistry() # dataflows
        print("augs_reg", augs_reg._reg)
        print("flows_reg", flow_reg._reg)
        need_remove_custom_params = False
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
                if ds_imgs.is_add_custom_params():
                    need_remove_custom_params = True
            else:
                print("WARNING: unsuported augmetator: ", name, "skiped...")
        if need_remove_custom_params:
            print("Create RemoveCustomParamsFlow")
            ds_imgs = RemoveCustomParamsFlow(ds_imgs)
    if cfg.workers:
        print("Starting %i workers for fetch data with size %i..." % (cfg.workers, ds_imgs.size()))
        ds_imgs = PrefetchDataZMQ(ds_imgs, nr_proc=cfg.workers)
        if endless:
            ds_imgs._size = -1 # for endless loop # HACK

    if cfg.shuffle_buffer:
        ds_imgs = LocallyShuffleData(ds_imgs, cfg.shuffle_buffer)
        if endless:
            ds_imgs.size = lambda : 2**64 # HACK
    return ds_imgs
