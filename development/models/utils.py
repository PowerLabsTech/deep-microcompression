import random
import itertools
import copy
import logging
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, Optional, Callable, Union

import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn

from .sequential import Sequential
from ..layers import *


def convert_from_sequential_torch_to_dmc(torch_model: nn.Sequential) -> Sequential:
    """
    Converts a pytorch Sequential model to its DMC equivalent, if there is an equivalent
    implementation for all the layers

    :param: torch_model: pytorch sequential model

    :param: dmc_model: DMC eqivalent model with all of its layers converted
    """
    assert isinstance(torch_model, (nn.Sequential or Sequential)), "Model must be a torch sequential model."

    dmc_model = Sequential()
    for module in torch_model:
        dmc_model = dmc_model + convert_from_layer_torch_nn_to_dmc(module)
    return dmc_model


def convert_from_layer_torch_nn_to_dmc(module: nn.Module) -> Layer:
    """
    Converts a pytorch module to its equivalent DMC layer. It is only successful
    for some modules but this list is growing fast. Feel free to contribute.

    :param: module: pytorch module to be converted to its dmc equivalent

    :return: layer: DMC equivalent of module
    """
    @torch.no_grad()
    def copy_tensor(tensor_source, tensor_destination):
        tensor_destination.copy_(tensor_source)

    if isinstance(module, nn.Conv2d):
        padding = [module.padding[0]]*2 + [module.padding[1]]*2
        if module.padding_mode == "zeros":
            value = 0
        else:
            raise ValueError(f"Padding Layer to handle padding_mode {module.padding_mode} is yet to be defined.")
        
        layer = Conv2d(
            in_channels=module.in_channels, out_channels=module.out_channels, 
            kernel_size=module.kernel_size, stride=module.stride,
            bias=module.bias is not None
        )
        copy_tensor(module.weight, layer.weight)
        if layer.bias is not None:
            copy_tensor(module.bias, layer.bias)

        layer = Sequential(
            ConstantPad2d(padding=padding, value=value),
            layer
        )
    elif isinstance(module, nn.Linear):
        layer = Linear(
            in_features=module.in_features, out_features=module.out_features, 
            bias=module.bias is not None
        )
        copy_tensor(module.weight, layer.weight)
        if layer.bias is not None:
            copy_tensor(module.bias, layer.bias)

    elif isinstance(module, nn.ReLU):
        layer = ReLU(inplace=module.inplace) 

    elif isinstance(module, nn.ReLU6):
        layer = ReLU6(inplace=module.inplace) 


    elif isinstance(module, nn.Flatten):
        layer = Flatten(start_dim=module.start_dim, end_dim=module.end_dim) 
        
    elif isinstance(module, nn.MaxPool2d):
        layer = MaxPool2d(
            module.kernel_size, module.stride,
            module.padding, module.dilation,
            ceil_mode=module.ceil_mode, return_indices=module.return_indices
        )

    elif isinstance(module, nn.AvgPool2d):
        layer = AvgPool2d(
            module.kernel_size, module.stride,
            module.padding, module.dilation,
            ceil_mode=module.ceil_mode, return_indices=module.return_indices
        )
        
    elif isinstance(module, nn.BatchNorm2d):
        
        layer = BatchNorm2d(num_features=module.num_features, eps=module.eps, momentum=module.momentum, affine=module.affine,  track_running_stats=module.track_running_stats)
        if module.affine:
            copy_tensor(module.weight, layer.weight)
            copy_tensor(module.bias, layer.bias)
        if module.track_running_stats:
            copy_tensor(module.running_mean, layer.running_mean)
            copy_tensor(module.running_var, layer.running_var)

    elif isinstance(module, nn.Dropout):
        return Dropout(p=module.p, inplace=module.inplace)
    else:
        raise RuntimeError(f"module of type {type(module)} does not have a dmc equivalent yet.")


    return layer


def sample_nas_compression_configs(
    model: Sequential,
    input_shape: Tuple,
    calibration_data: torch.Tensor,
    n_configs: int = 1000,
    filter: Callable = lambda compressed_model, compression_config: True,
    device: str = "cpu",
    random_seed: Optional[int] = None,
    deduplicate: bool = True,
    max_attempts: int = 200_000,
) -> Tuple[List[Dict], int]:
    """
    Pre-generate a pool of hardware-valid compression configs in one pass.

    Returns raw (encoded) config dicts. Pass a slice to get_nas_compression_data()
    via the `configs` argument to guarantee zero duplicate configs across parallel workers.

    Args:
        n_configs:    Number of unique valid configs to generate.
        filter:       Callable(compressed_model, decoded_config) -> bool.
        deduplicate:  Reject configs with duplicate hyperparameter combinations.
        max_attempts: Hard cap on random draws before raising RuntimeError.

    Returns:
        List of raw (encoded) config dicts, length n_configs.
        Number of attempts to generate the n_configs
    """
    if random_seed is not None:
        random.seed(random_seed)

    fused_model             = model.fuse()
    compression_possible_hp = fused_model.get_compression_possible_hyperparameters()

    pool     = []
    seen     = set()
    attempts = 0

    with tqdm(total=n_configs, desc="Sampling configs") as pbar:
        while len(pool) < n_configs:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Could not generate {n_configs} unique valid configs after {max_attempts} "
                    f"attempts. Found {len(pool)}. The filter may be too restrictive."
                )

            compression_config = {
                config_key: random.choice(list(config_hp))
                for config_key, config_hp in compression_possible_hp.items()
            }
            if deduplicate:
                key = str(sorted(compression_config.items()))
                if key in seen:
                    continue

            decoded = fused_model.decode_compression_dict_hyperparameter(compression_config)
            if not fused_model.is_compression_config_valid(
                decoded, raise_error=False
            ):
                continue

            compressed = fused_model.init_compress(
                decoded, input_shape, calibration_data=calibration_data, device=device
            )
            if not filter(compressed, decoded):
                continue

            if deduplicate:
                seen.add(key)
            pool.append(compression_config)
            pbar.update(1)
            pbar.set_postfix(attempts=attempts, hit_rate=f"{len(pool)/attempts*100:.1f}%")

    return pool, attempts


def get_nas_compression_data(
    model:Sequential,
    input_shape:Tuple,
    data_loader:torch.utils.data.DataLoader,
    metric_fun:Callable[[torch.Tensor, torch.Tensor], float],
    calibration_data:torch.Tensor,
    filter:Callable = lambda compressed_model, compression_config: True,
    device:str="cpu",
    num_data:int=100,
    configs:Optional[List[Dict]] = None,
    train:bool=False,
    train_dataloader:Optional[torch.utils.data.DataLoader] = None,
    epochs:Optional[int] = None,
    criterion_fun:Optional[nn.Module] = None,
    lr:float=1e-3,
    optimizer_cls:Optional[torch.optim.Optimizer] = None,
    optimizer_kwargs:Optional[Dict] = None,
    lr_scheduler_cls:Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    lr_scheduler_kwargs:Optional[Dict] = None,
    callbacks:List[Callable] = [],
    random_seed:Optional[int] = None,
    two_step:bool=True,
    eval_subset_size:Optional[int] = None,
    max_filter_attempts:int = 200_000,
    checkpoint_path:Optional[str] = None
) -> Dict[str, List]:
    """
    Fine-tune and evaluate compression configs, returning a NAS training dataset.

    Two modes:
    - **Pool mode** (``configs`` provided): iterates the given list of raw config dicts.
      Use a slice from :func:`sample_nas_compression_configs` for guaranteed uniqueness
      across parallel workers — no random sampling is done.
    - **Random mode** (``configs=None``): samples ``num_data`` random valid configs on
      the fly (original behaviour).

    Args:
        configs:           Pre-generated raw config dicts from sample_nas_compression_configs().
                           When provided, ``num_data`` and ``random_seed`` are ignored for
                           sampling purposes.
        num_data:          Number of random configs to sample (only used when configs=None).
        eval_subset_size:  Evaluate on a random subset of data_loader for speed.
        max_filter_attempts: Max draws per sample before raising RuntimeError (random mode only).

    Returns:
        Dict mapping compression parameter names (and "metric") to lists of values.
    """
    if random_seed is not None and configs is None:
        random.seed(random_seed)

    if train:
        assert train_dataloader is not None, "if training you must pass train_dataloader"
        assert epochs is not None,           "if training you must pass epochs"
        assert criterion_fun is not None,    "if training you must pass criterion_fun"

    fused_model = model.fuse()

    # --- build a fast eval loader (subset) for NAS metric estimation ---
    if eval_subset_size is not None and eval_subset_size < len(data_loader.dataset):
        subset_indices = random.sample(range(len(data_loader.dataset)), eval_subset_size)
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(data_loader.dataset, subset_indices),
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=getattr(data_loader, "num_workers", 0),
            pin_memory=getattr(data_loader, "pin_memory", False),
        )
    else:
        eval_loader = data_loader

    compression_config_data = defaultdict(list)

    # --- decide iteration source ---
    if configs is None:
        configs, _ = sample_nas_compression_configs(
            model=model,
            input_shape=input_shape,
            calibration_data=calibration_data,
            n_configs=num_data,
            filter=filter,
            device=device,
            random_seed=random_seed,
            deduplicate=True,
            max_attempts=max_filter_attempts,
        )

    pbar = tqdm(configs, total=len(configs), desc="NAS data")
    for item in pbar:

        # item is already a raw config dict
        compression_config        = item
        compression_config_decode = fused_model.decode_compression_dict_hyperparameter(compression_config)

        # --- optional fine-tuning ---
        if train:
            prune_config    = {"prune_channel": compression_config_decode["prune_channel"]}
            quantize_config = {"quantize": compression_config_decode["quantize"]}

            pbar.set_postfix(phase="pruning")
            pruned_model = fused_model.init_compress(
                prune_config, input_shape,
                calibration_data=calibration_data, device=device,
            )

            if two_step:
                prune_epochs = epochs // 2
                quant_epochs = epochs - prune_epochs

                pbar.set_postfix(phase=f"prune-train ({prune_epochs}ep)")
                optimizer_fun = optimizer_cls(pruned_model.parameters(), lr=lr, **optimizer_kwargs)
                lr_scheduler  = lr_scheduler_cls(optimizer_fun, **lr_scheduler_kwargs)
                pruned_model.fit(
                    train_dataloader=train_dataloader,
                    epochs=prune_epochs,
                    criterion_fun=criterion_fun,
                    optimizer_fun=optimizer_fun,
                    lr_scheduler=lr_scheduler,
                    validation_dataloader=eval_loader,
                    metrics={"metric": metric_fun},
                    verbose=False,
                    progress=False,
                    callbacks=callbacks,
                    device=device,
                )

            else:
                quant_epochs = epochs

            pbar.set_postfix(phase="quantizing")
            compressed_model = pruned_model.init_compress(
                config=quantize_config,
                input_shape=input_shape,
                calibration_data=calibration_data,
                device=device,
            )

            pbar.set_postfix(phase=f"quant-train ({quant_epochs}ep)")
            optimizer_fun = optimizer_cls(compressed_model.parameters(), lr=lr, **optimizer_kwargs)
            lr_scheduler  = lr_scheduler_cls(optimizer_fun, **lr_scheduler_kwargs)
            compressed_model.fit(
                train_dataloader=train_dataloader,
                epochs=quant_epochs,
                criterion_fun=criterion_fun,
                optimizer_fun=optimizer_fun,
                lr_scheduler=lr_scheduler,
                validation_dataloader=eval_loader,
                metrics={"metric": metric_fun},
                verbose=False,
                progress=False,
                callbacks=callbacks,
                device=device,
            )
        else:
            pbar.set_postfix(phase="compressing")
            compressed_model = fused_model.init_compress(
                config=compression_config_decode,
                input_shape=input_shape,
                calibration_data=calibration_data,
                device=device,
            )

        pbar.set_postfix(phase="evaluating")
        compressed_model_metric = compressed_model.evaluate(
            data_loader=eval_loader, metrics={"metric": metric_fun},
            device=device, progress=False
        )

        for config_key, config_value in compression_config.items():
            compression_config_data[config_key].append(config_value)
        compression_config_data["metric"].append(compressed_model_metric["metric"])

        if checkpoint_path is not None:
            torch.save(dict(compression_config_data), checkpoint_path)

    return compression_config_data


def get_size_of_compression(
    model:Sequential, 
    condition:Callable = lambda compression_config: True
):
    # helper to list all compression combinations
    num_valid_compressions = 0
    total_compressions = 0
    search_space = model.get_compression_possible_hyperparameters()
    config_keys = list(search_space.keys())
    config_values = list(search_space.values())
    for combo in tqdm(itertools.product(*config_values)):
        config = {config_key: config_value for config_key, config_value in zip(config_keys, combo)}
        decode_config = model.decode_compression_dict_hyperparameter(config)
        if model.is_compression_config_valid(
            decode_config, 
            compression_keys=["quantize"], 
            raise_error=False
        ):
            if condition(decode_config):
                num_valid_compressions += 1
            total_compressions += 1
    return num_valid_compressions, total_compressions


def brute_force_search_compression_config(
    model:Sequential,
    estimator:nn.Module,
    input_shape:Tuple,
    calibration_data:torch.Tensor,
    condition:Callable=lambda metric, size, ram, config: True,                 # list of conditions
    objective:Callable=lambda metric, size, ram, config: metric,                  # objective function, default objective = maximize metric
    maximize:bool=True,                   # maximize or minimize?
    verbose:bool=True
):
    """
    Generic brute-force search engine.

    Args:
        model: model with pruning methods
        estimator: metric estimator
        input_shape: input dimensions
        conditions: list of callables (metric,size,ram,comb)->bool
        objective: callable (metric,size,ram,comb)->float
        maximize: True=maximize objective, False=minimize
        verbose: print progress

    Returns:
        best_config, best_obj_value, history
    """

    # initialize best value based on maximize/minimize
    if maximize:
        best_value = float("-inf")
    else:
        best_value = float("inf")

    best_config = None

    search_space = model.get_compression_possible_hyperparameters()

    # compute baseline size

    # helper to list all compression combinations
    def get_all_combinations(search_space):
        config_keys = list(search_space.keys())
        config_values = list(search_space.values())
        for combo in itertools.product(*config_values):
            config = {config_key: config_value for config_key, config_value in zip(config_keys, combo)}
            if model.is_compression_config_valid(
                model.decode_compression_dict_hyperparameter(config), 
                compression_keys=["quantize"], 
                raise_error=False
            ):
                yield config

    # iterate search space
    for compression_config in get_all_combinations(search_space):
        # predict metric
        metric = estimator(compression_config)

        # create compressed model to get the model size
        compressed = model.init_compress(
            model.decode_compression_dict_hyperparameter(compression_config), 
            input_shape, calibration_data=calibration_data
        )

        size = compressed.get_size_in_bytes()
        ram = compressed.get_workspace_size(input_shape)

        # -------- HARD FILTERS --------
        if not condition(metric, size, ram, compression_config):
            continue

        # -------- OBJECTIVE VALUE --------
        obj = objective(metric, size, ram, compression_config)

        if (maximize and obj > best_value) or (not maximize and obj < best_value):
            best_value = obj
            best_config = compression_config
            best_result_info = [metric, size, ram]
            if verbose:
                print(f"✔ New best: obj={obj:.4f}, metric={metric:.4f}, size={size:.4f}, ram={ram}, comb={compression_config}")

    best_result_info = {
        "metric": best_result_info[0],
        "size": best_result_info[1],
        "ram": best_result_info[2]
    }
    return best_config, best_result_info


def evolutionary_search_compression_config(
    model:Sequential,
    estimator:nn.Module,
    input_shape:Tuple,
    calibration_data:torch.Tensor,
    device:str="cpu",
    condition:Callable=lambda metric, size, ram, config: True,                 # list of conditions
    objective:Callable=lambda metric, size, ram, config: metric,                  # objective function, default objective = maximize metric
    filter:Callable = lambda compressed_model, compression_config: True,
    maximize:bool=True,                   # maximize or minimize?
    verbose:bool=True,
    population_size:int=50,
    generations:int=50,
    mutation_rate:float=0.2,
    crossover_rate:float=0.7,
    elite_size:int=2,
    turnament_size:int=3
):
    """
    Genetic Algorithm search engine for optimal pruning configuration.
    """

    # Analyze Search Space
    search_space = model.get_compression_possible_hyperparameters()

    calibration_data = calibration_data.to(device=device)
    
    param_names = list(search_space.keys())
    param_values = list(search_space.values()) # List of lists of valid values
    
    # Pre-compute baseline stats

    def create_individual():
        """Generate a random individual model compression"""
        valid = False
        while not valid:
            config = {config_key: random.choice(config_values) for config_key, config_values in zip(param_names, param_values)}
            compression_config_decode = model.decode_compression_dict_hyperparameter(config)
            valid = model.is_compression_config_valid(
                compression_config_decode, 
                compression_keys=["quantize"], 
                raise_error=False
            )
            if valid:
                compressed_model = model.init_compress(
                    config=compression_config_decode,
                    input_shape=input_shape, calibration_data=calibration_data,
                    device=device
                )

                valid = filter(compressed_model, compression_config_decode)
        return config

    def evaluate(compression_config):
        """Evaluates a model to get its metric based on the estimator"""

        if not model.is_compression_config_valid(
            model.decode_compression_dict_hyperparameter(compression_config), 
            compression_keys=["quantize"], 
            raise_error=False
        ):
            return (float("-inf") if maximize else float("inf")), [None, None, None]
        
        metric = estimator(compression_config)

        compressed = model.init_compress(
            model.decode_compression_dict_hyperparameter(compression_config),
            input_shape, calibration_data=calibration_data, device=device
        )

        size = compressed.get_size_in_bytes()
        ram = compressed.get_workspace_size(input_shape)

        # -------- HARD FILTERS --------
        if not condition(metric, size, ram, model.decode_compression_dict_hyperparameter(compression_config)):
            # Penalty fitness for invalid solutions
            return (float("-inf") if maximize else float("inf")), [metric, size, ram]

        # -------- OBJECTIVE VALUE --------
        score = objective(metric, size, ram, compression_config)
        return score, [metric, size, ram]

    def crossover(parent1, parent2):
        """Creates a model config by randomly crossing over two parent models"""
        child = dict()
        for key in param_names:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate(individual):
        """Creates a model config by randomly mutating a model"""
        mutated = individual.copy()
        # Pick a random gene (layer) to mutate
        gene_to_mutate = random.choice(param_names)
        # Pick a new valid value from the search space for that layer
        valid_options = search_space[gene_to_mutate]
        mutated[gene_to_mutate] = random.choice(valid_options)
        return mutated

    # Evolutionary Loop
    population = [create_individual() for _ in range(population_size)]
    
    best_overall_config = None
    best_overall_score = float("-inf") if maximize else float("inf")
    best_overall_info = []

    for gen in tqdm(range(generations)):
        # Evaluate Population
        scores = []
        infos = []
        for individual in population:
            score, info = evaluate(individual)
            scores.append(score)
            infos.append(info)
            
            # Update Global Best
            is_better = (score > best_overall_score) if maximize else (score < best_overall_score)
            if is_better and score != float("-inf") and score != float("inf"):
                best_overall_score = score
                best_overall_config = individual
                best_overall_info = info
                if verbose:
                    print(f"Gen {gen}: ✔ New best! Obj={score:.4f} Metric={info[0]:.4f} Size={info[1]:.4f} Ram={info[2]:.4f}")

        # Selection Pool
        selected = []
        while len(selected) < population_size:
            # Pick k random candidates
            candidates_indices = random.sample(range(population_size), k=turnament_size)
            # Find the best among them
            best_idx = candidates_indices[0]
            for idx in candidates_indices[1:]:
                if maximize:
                    if scores[idx] > scores[best_idx]: best_idx = idx
                else:
                    if scores[idx] < scores[best_idx]: best_idx = idx
            selected.append(population[best_idx])

        # Elitism (Keep best individuals)
        # Sort population by score
        sorted_pop_indices = np.argsort(scores)
        if maximize:
            sorted_pop_indices = sorted_pop_indices[::-1] # Descending
            
        next_generation = []
        for i in range(elite_size):
            idx = sorted_pop_indices[i]
            # Only keep if valid
            if scores[idx] != float("inf") and scores[idx] != float("-inf"):
                next_generation.append(population[idx])

        # 5. Crossover & Mutation
        while len(next_generation) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1 # No crossover, just copy
            
            if random.random() < mutation_rate:
                child = mutate(child)
                
            next_generation.append(child)
            
        population = next_generation
        
        if verbose:
            valid_scores = [s for s in scores if s != float("inf") and s != float("-inf")]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
            print(f"Gen {gen}/{generations} Complete. Avg Score: {avg_score:.4f}")
    try:
        metric = best_overall_info[0]
        size = best_overall_info[1]
        ram = best_overall_info[2]

        best_overall_info = {
            "metric": metric,
            "size": size,
            "ram": ram
        }
        return best_overall_config, best_overall_info
    
    except IndexError:
        return None, [None, None, None]

                
        