import random
import itertools
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, Optional, Callable

import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn

from .estimator import Estimator
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
        pad = [module.padding[0]]*2 + [module.padding[1]]*2
        layer = Conv2d(
            in_channels=module.in_channels, out_channels=module.out_channels, 
            kernel_size=module.kernel_size, stride=module.stride, pad=pad, 
            bias=module.bias is not None
        )
        copy_tensor(module.weight, layer.weight)
        if layer.bias is not None:
            copy_tensor(module.bias, layer.bias)

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
        return ReLU()
    else:
        raise RuntimeError(f"module of type {type(module)} does not have a dmc equivalent yet.")


    return layer


def get_nas_compression_data(
    model:Sequential,
    input_shape:Tuple, 
    data_loader:torch.utils.data.DataLoader, 
    metric_fun:Callable[[torch.Tensor, torch.Tensor], float], 
    calibration_data:torch.Tensor,
    device:str="cpu",
    num_data:int=100,
    train:bool=False,
    train_dataloader:Optional[torch.utils.data.DataLoader] = None,
    epochs:Optional[int] = None,
    criterion_fun:Optional[nn.Module] = None,
    lr_scheduler:Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    callbacks:List[Callable] = [],
    random_seed:Optional[int] = None,
) -> Dict[str, List]:
    """
    Generates data for Neural Architecture Search (NAS) / Random Search.
    It randomly samples valid configuration to explore the global sparsity space.

    :parma: num_data: Number of random architectures to sample
    
    :return: List of configurations and result [param_1, param_2, ..., param_n, accuracy] vectors.
    """

    if random_seed: random.seed(random_seed)
    
    if train:
        assert train_dataloader is not None, f"if training you have to pass train_dataloader"
        assert epochs is not None, f"if training you have to pass epochs"
        assert criterion_fun is not None, f"if training you have to pass criterion_fun"

    compression_config_data = defaultdict(list)
    
    compression_possible_hp = model.get_commpression_possible_hyperparameters()
    for _ in range(num_data):
        # generating a random configuration
        valid = False
        while not valid:
            compression_config = {config_key: random.choice(config_hp) for config_key, config_hp in compression_possible_hp.items()}
            valid = model.is_compression_config_valid(
                model.decode_compression_dict_hyperparameter(compression_config),
                compression_keys=["quantize"],
                raise_error=False
            )

        compressed_model = model.init_compress(
            config=model.decode_compression_dict_hyperparameter(compression_config),
            input_shape=input_shape, calibration_data=calibration_data
        )
        
        if train:
            optimizer_fun = torch.optim.SGD(compressed_model.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=1)

            compressed_model.fit(
                train_dataloader=train_dataloader,
                epochs=epochs,
                criterion_fun=criterion_fun,
                optimizer_fun=optimizer_fun,
                lr_scheduler=lr_scheduler,
                validation_dataloader=data_loader,
                metrics={"metric": metric_fun},
                verbose=False,
                callbacks=callbacks,
                device=device,
            )

            compressed_model_metric = compressed_model.evaluate(
                data_loader=data_loader, metrics={"metric": metric_fun}, 
                device=device
            )
        else:
            compressed_model_metric = compressed_model.evaluate(
                data_loader=data_loader, metrics={"metric": metric_fun}, 
                device=device
            )

        for config_key, config_value in compression_config.items():
            compression_config_data[config_key].append(config_value)
        compression_config_data["metric"].append(compressed_model_metric["metric"])

    return compression_config_data


def brute_force_search_compression_config(
    model:Sequential,
    estimator:Estimator,
    input_shape:Tuple,
    calibration_data:torch.Tensor,
    condition:Callable=lambda metric, size, ram, config: True,                 # list of filters
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
        best_comb, best_obj_value, history
    """

    # initialize best value based on maximize/minimize
    if maximize:
        best_value = float("-inf")
    else:
        best_value = float("inf")

    best_comb = None

    search_space = model.get_commpression_possible_hyperparameters()

    # compute baseline size
    original_size = model.get_size_in_bytes()
    original_metric = estimator.predict({
        config_key: config_values[0] for config_key, config_values in search_space.items()
    })

    # helper to list all prune combinations
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
        metric = estimator.predict(compression_config) / original_metric

        # create compressed model to get the model size
        compressed = model.init_compress(
            model.decode_compression_dict_hyperparameter(compression_config), 
            input_shape, calibration_data=calibration_data
        )

        size = compressed.get_size_in_bytes() / original_size
        ram = sum(compressed.get_max_workspace_arena(input_shape)) / 2

        # -------- HARD FILTERS --------
        if not condition(metric, size, ram, compression_config):
            continue

        # -------- OBJECTIVE VALUE --------
        obj = objective(metric, size, ram, compression_config)

        if (maximize and obj > best_value) or (not maximize and obj < best_value):
            best_value = obj
            best_comb = compression_config
            best_result_info = [metric, size, ram]
            if verbose:
                print(f"✔ New best: obj={obj:.4f}, metric={metric:.4f}, size={size:.4f}, ram={ram}, comb={compression_config}")

    return best_comb, best_result_info


def evolutionary_search_compression_config(
    model:Sequential,
    estimator:Estimator,
    input_shape:Tuple,
    calibration_data:torch.Tensor,
    condition:Callable=lambda metric, size, ram, config: True,                 # list of filters
    objective:Callable=lambda metric, size, ram, config: metric,                  # objective function, default objective = maximize metric
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
    search_space = model.get_commpression_possible_hyperparameters()
    
    param_names = list(search_space.keys())
    param_values = list(search_space.values()) # List of lists of valid values
    
    # Pre-compute baseline stats
    original_size = model.get_size_in_bytes()
    original_metric = estimator.predict({
        config_key: config_values[0] for config_key, config_values in zip(param_names, param_values)
    })

    def create_individual():
        """Generate a random individual model compression"""
        valid = False
        while not valid:
            config = {config_key: random.choice(config_values) for config_key, config_values in zip(param_names, param_values)}
            valid = model.is_compression_config_valid(
                model.decode_compression_dict_hyperparameter(config), 
                compression_keys=["quantize"], 
                raise_error=False
            )
        return config

    def evaluate(compression_config):
        """Evaluates a model to get its metric based on the estimator"""

        if not model.is_compression_config_valid(
            model.decode_compression_dict_hyperparameter(compression_config), 
            compression_keys=["quantize"], 
            raise_error=False
        ):
            return (float("-inf") if maximize else float("inf")), [None, None, None]
            

        metric = estimator.predict(compression_config) / original_metric
        compressed = model.init_compress(
            model.decode_compression_dict_hyperparameter(compression_config),
            input_shape, calibration_data=calibration_data
        )

        size = compressed.get_size_in_bytes() / original_size
        ram = sum(compressed.get_max_workspace_arena(input_shape)) / 2

        # -------- HARD FILTERS --------
        if not condition(metric, size, ram, compression_config):
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
    
    best_overall_comb = None
    best_overall_score = float("-inf") if maximize else float("inf")
    best_overall_info = None

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
                best_overall_comb = individual
                best_overall_info = info
                if verbose:
                    print(f"Gen {gen}: ✔ New best! Obj={score:.4f} Metric={info[0]:.4f} Size={info[1]:.4f}")

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
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            print(f"Gen {gen}/{generations} Complete. Avg Score: {avg_score:.4f}")

    return best_overall_comb, best_overall_info
                
        