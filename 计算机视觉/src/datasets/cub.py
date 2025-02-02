from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
import os

def get_CUB200(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns MiniImagenet datasets.
    """
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        
        if args.dataset_version not in [None, "2011"]:
            raise Exception("Dataset version not found {}".format(args.dataset_version))
            
        data_path = os.path.abspath(args.data_path)
        
        if args.dataset_version in [None, "2011"]:
            filepath = os.path.join(data_path, "cub", "cub-cache-{0}.pkl".format(setname))
            data = load_dataset_from_pkl(filepath)
            dataset_class = ColorDatasetInMemory
            
        datasets[setname] = [data['image_data'], data['class_dict'], args, dataset_class]
        
    return datasets