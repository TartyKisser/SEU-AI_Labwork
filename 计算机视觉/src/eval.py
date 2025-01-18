import numpy as np
import tqdm
import os
import copy
import pprint
from dataclasses import dataclass
from typing import Dict, List
from utils.ptracker import PerformanceTracker
from tasks.task_generator import TaskGenerator
from utils.dataloader import DataLoader
from demo_fsl_task import DemoFSLTask
from utils.bunch import bunch
from utils.builder import ExperimentBuilder
from utils.utils import set_torch_seed, set_gpu, get_tasks, get_data, get_model, get_backbone, get_strategy, get_args, compress_args

@dataclass
class ModelBuilder:
    """Model wrapper for evaluation."""
    model: any
    datasets: Dict[str, any]
    task_args: Dict[str, any]
    device: str
    state: any
    args: any

import csv  # 导入CSV库

def evaluate_models(systems: Dict[str, ModelBuilder],
                    supports_idx: List[int],
                    targets_idx: List[int],
                    ptracker_args: Dict[str, any]) -> Dict[str, any]:
    """
    Evaluate trained models on a given support and target set.

    Args:
        systems (Dict[str, ModelBuilder]): A dictionary mapping model names to ModelBuilder objects.
        supports_idx (List[int]): Indexes of support set samples.
        targets_idx (List[int]): Indexes of target set samples.
        ptracker_args (Dict[str, any]): Performance tracker configuration.

    Returns:
        Dict[str, any]: A dictionary containing predictions, ground truth labels, and support set labels for each model.
    """
    data = {'action': 'output', 'models': {}}

    with tqdm.tqdm(total=len(systems), disable=False) as pbar_val:
        for model_name, builder in systems.items():
            # Initialize performance tracker for this model
            ptracker = PerformanceTracker(args=ptracker_args)
            print("Available keys in inv_class_dict:", builder.datasets['test'].inv_class_dict.keys())
            print("Supports Indexes:", supports_idx)

            # Map support and target indices to labels
            supports_lblname = [builder.datasets['test'].inv_class_dict[i] for i in supports_idx]
            targets_lblname = [builder.datasets['test'].inv_class_dict[i] for i in targets_idx]

            # Generate unique label mappings for supports
            slbl_uniq, supports_lbl = np.unique(supports_lblname, return_inverse=True)
            tlbl_uniq = np.array(slbl_uniq.tolist())
            tlbl_uniq_map = {n: i for i, n in enumerate(tlbl_uniq)}
            targets_lbl = np.array([tlbl_uniq_map[name] for name in targets_lblname])

            # Prepare task arguments
            task_args = {"test": {"support_idx": supports_idx, "support_lbls": supports_lbl,
                                  "target_idx": targets_idx, "target_lbls": targets_lbl}}

            print(f"Testing {model_name} with {len(supports_idx)} supports and {len(targets_idx)} targets.")

            # Configure the model and task
            ptracker.set_mode('test')
            builder.model.set_mode('test')
            builder.task_args['test'] = task_args['test']

            # Generate tasks and evaluate
            task_generator = TaskGenerator(
                builder.datasets['test'],
                task=DemoFSLTask,
                task_args=task_args['test'],
                num_tasks=1,
                seed=builder.args.seed,
                epoch=builder.state.epoch,
                mode='test',
                fix_classes=False,
                deterministic=True
            )

            for sampler in task_generator:
                dataloader = DataLoader(
                    builder.datasets['test'],
                    sampler,
                    builder.device,
                    builder.state.epoch,
                    'test'
                )
                builder.model.meta_test(dataloader, ptracker)

            # Update progress bar and log performance
            pbar_val.set_description(f"Testing ({model_name}) -> {ptracker.get_performance_str()}")

            # 获取真实标签和预测结果
            preds = ptracker.lastest_task_performance["preds"]  # 预测标签

            # 映射预测结果和真实标签到标签名称
            predicted_lbls = [tlbl_uniq[pred] for pred in preds]
            real_labels = targets_lblname  # 目标集的真实标签名称
            support_labels = supports_lblname  # 支持集的标签名称

            # 保存到输出数据
            data['models'][model_name] = {
                "support_labels": support_labels,   # 支持集标签名称
                "real_labels": real_labels,         # 目标集真实标签名称
                "predicted_labels": predicted_lbls  # 预测标签名称
            }

    return data





if __name__ == "__main__":
    import argparse
    import json

    # Example usage
    parser = argparse.ArgumentParser(description="Evaluate trained models on given tasks.")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Example model system loading
    # Here, you need to define how models and datasets are loaded
    loaded_models = {}
    datasets = None
    abspath = os.path.abspath("..")
    print(args)

    models_config = config.get("models", [])  # 获取 models，如果不存在返回空列表
    seed = config.get("seed", None)
    dataset = config.get("dataset", None)
    version = config.get("version", None)
    data_path = config.get("data_path", None)
    for builder_args in models_config:
        original_args = copy.copy(builder_args)

        assert 'continue_from' in builder_args, 'All "models" should have a "continue_from" entry.'
        assert 'gpu' in builder_args, 'All "models" should have a specified "gpu" entry or "cpu" device.'

        stdin_list = [
            "--args_file", os.path.join(abspath,'experiments/imbalanced_supports/cub/Conv4', builder_args["continue_from"], 'configs', 'config.json'),
            "--continue_from", os.path.join(abspath, 'experiments/imbalanced_supports/cub/Conv4',builder_args["continue_from"]),
            "--gpu", builder_args['gpu'],
            "--seed", seed,
            "--dataset", dataset,
            "--dataset_args", json.dumps({'dataset_version': version,
                                          'data_path': data_path})
        ]

        builder_args, excluded_args, parser = get_args(stdin_list)
        builder_args = bunch.bunchify(builder_args)

        compressed_args = compress_args(bunch.unbunchify(builder_args), parser)

        device = set_gpu(builder_args.gpu)
        tasks = get_tasks(builder_args)
        datasets = get_data(builder_args) if datasets is None else datasets
        backbone = get_backbone(builder_args, device)
        strategy = get_strategy(builder_args, device)
        model = get_model(backbone, tasks, datasets, strategy, builder_args, device)

        compressed_args = compress_args(bunch.unbunchify(builder_args), parser)
        print(" ----------------- FULL ARGS (COMPACT) ----------------")
        pprint.pprint(compressed_args, indent=2)
        print(" ------------------------------------------------------")
        print(" ------------------ UNRECOGNISED ARGS -----------------")
        pprint.pprint(excluded_args, indent=2)
        print(" ------------------------------------------------------")

        system = ExperimentBuilder(model, tasks, datasets, device, builder_args)
        system.load_pretrained()

        model.set_mode('test')

    # Example: Supports and targets
    # 从 config 中正确读取 supports_idx 和 targets_idx
    supports_idx = config.get("task_args", {}).get("test", {}).get("support_idx", [])
    targets_idx = config.get("task_args", {}).get("test", {}).get("target_idx", [])

    #ptracker_args = builder_args["ptracker_args"]
    ptracker_args = config.get("ptracker_args", {})
    # Evaluate models
    systems = {"model1": system}  # 包装成字典
    results = evaluate_models(systems, supports_idx, targets_idx, ptracker_args)

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
