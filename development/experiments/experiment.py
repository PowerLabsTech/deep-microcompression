import os
import argparse
import logging
import datetime
import time
import copy

import torch
from torch import nn, optim

# import sys
# sys.path.append("../..")

from .. import (
    Sequential,
    EarlyStopper,
    QuantizationScheme,
    # Estimator,
    get_nas_compression_data,
    brute_force_search_compression_config, 
    evolutionary_search_compression_config,
    get_size_of_compression
)

# import warnings
# warnings.filterwarnings('ignore')


KB = 1024
ram = 2 * KB
flash = 32 * KB 
VERTEXAI_ENV = "vertexai"
GCS_BUCKET = "deep-microcompression-496018"

experiment_args_file = "experiment_args.pth"
baseline_model_file = "baseline_model.pth"
best_configuration_smallest_metric_file = "best_configuration_smallest_metric_file.pth"
best_configuration_smallest_for_board_file = "best_configuration_smallest_for_board_file.pth"
estimator_training_history_file = "estimator_training_history_file.pth"
nas_param_file = "nas_parameters.pth"
root_dir = "/home/matthias/Documents/EmbeddedAI/deep-microcompression/development"
 

def save_file(object, local_filename, overwrite=True):

    if experiment_args.env == VERTEXAI_ENV:
        storage_filename = os.path.join(f"experiments/{experiment_args.experiment_name}", local_filename)
        local_filename = os.path.join("/tmp", local_filename)
        
        torch.save(object, local_filename)

        from google.cloud import storage
        bucket = storage.Client().get_bucket(f"{GCS_BUCKET}_{experiment_args.LOCATION}")

        blob = bucket.blob(storage_filename)
        if blob.exists() and not overwrite:
            return
        
        blob.upload_from_filename(local_filename)
        logger.info(f"Uploaded {local_filename} to {storage_filename}")

    else:
        dir_path = f"{root_dir}/experiments/{experiment_args.experiment_name}"
        os.makedirs(dir_path, exist_ok=True)
        local_filename = os.path.join(dir_path, local_filename)
        torch.save(object, local_filename)


def set_random_seed(seed):
    # Reproducibility
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_filter_for_nas(baseline_model, input_shape):

    original_workspace_size = baseline_model.get_workspace_size(input_shape=input_shape)
    original_model_size = baseline_model.get_size_in_bytes()

    def model_filter_for_nas(compressed_model:Sequential, compression_config):
        # filter out not static quantization
        if compression_config["quantize"]["scheme"] != QuantizationScheme.STATIC:
            return False
        # filter out if activation cannot size in ram
        if compressed_model.get_workspace_size(input_shape=input_shape) / original_workspace_size > experiment_args.search_min_flash_ratio:
            return False
        # filter out if model size cannot size in flash
        if compressed_model.get_size_in_bytes() / original_model_size > experiment_args.search_min_flash_ratio:
            return False
        return True
    
    return model_filter_for_nas


def get_search_for_smallest_best_condition(baseline_model, input_shape):

    original_workspace_size = baseline_model.get_workspace_size(input_shape=input_shape)
    original_model_size = baseline_model.get_size_in_bytes()

    def condition(metric, size, ram, config):
        # filter out not static quantization
        if config["quantize"]["scheme"] != QuantizationScheme.STATIC:
            return False
        # filter out if activation cannot size in ram
        if ram / original_workspace_size > experiment_args.search_min_flash_ratio:
            return False
        # filter out if model size cannot size in flash
        if size / original_model_size > experiment_args.search_min_flash_ratio:
            return False
        if original_metric - metric > experiment_args.max_metric_difference:
            return False
        return True
    
    return condition


# def train_compressed_model(
#         baseline_model, 
#         epoch, 
#         train_loader, 
#         test_loader, 
#         compression_config, 
#         metric, 
#         input_shape, 
#         lr,
#         lr_scheduler_patience,
#         lr_factor,
#         device, 
#         two_step=True
#     ):
#     logger.info("--- Applying Compression & Retraining ---")
    
#     logger.info(f"Applying compression configuration: {compression_config}")

#     calibration_data = next(iter(train_loader))[0].to(device=device)
    
#     if two_step:
#         prune_epoch = int(epoch/2)
#         epoch = epoch - prune_epoch

#         prune_config = copy.deepcopy(compression_config)
#         del prune_config["quantize"]
#         pruned_model = baseline_model.init_compress(prune_config, input_shape, calibration_data=calibration_data, device=device)

#         # Retrain (fine-tune) the compressed model
#         logger.info(f"Retraining compressed model in a single step approach ({epoch} epochs)...")
#         criterion_fun = nn.CrossEntropyLoss()
#         optimizer_fun = optim.SGD(pruned_model.parameters(), lr=lr, weight_decay=5e-4, momentum=.9)
#         lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=lr_scheduler_patience, factor=lr_factor)

#         pruned_model.fit(
#             train_loader, prune_epoch, 
#             criterion_fun, optimizer_fun, lr_scheduler,
#             validation_dataloader=test_loader, 
#             metrics={"metric": metric},
#             device=device
#         )

#         baseline_model = pruned_model

#     # Re-initialize model architecture with pruning
#     compressed_model = baseline_model.init_compress(compression_config, input_shape, calibration_data=calibration_data, device=device)
    
#     # Retrain (fine-tune) the compressed model
#     logger.info(f"Retraining compressed model in a single step approach ({epoch} epochs)...")
#     criterion_fun = nn.CrossEntropyLoss()
#     optimizer_fun = optim.SGD(compressed_model.parameters(), lr=lr, weight_decay=5e-4, momentum=.9)
#     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=lr_scheduler_patience, factor=lr_factor)

#     compressed_model.fit(
#         train_loader, epoch, 
#         criterion_fun, optimizer_fun, lr_scheduler,
#         validation_dataloader=test_loader, 
#         metrics={"metric": metric},
#         device=device
#     )

#     return compressed_model


def get_experiment_args():
    parser = argparse.ArgumentParser(
        prog="DMC model compression experimentts.",
        description=(f"It performs automated search for the best model compression"
                    f" configuration.")
    )

    parser.add_argument(
        "--seed", default=25, type=int, required=False,
        help="The experiment random seed for reproducibility."
    )

    parser.add_argument(
        "--experiment_name", "--exp", type=str,choices=["lenet5_mnist", "vgg13_cifar100", "resnet_cifar100"],
        required=True, help="The name of the model experiment you want to perform."
    )

    parser.add_argument(
        "--input_shape", required=True, type=int, nargs=3,
        help="The input shape the model expects."
    )

    # parser.add_argument(
    #     "--train", type=bool, action=argparse.BooleanOptionalAction, required=True,
    #     help="Determines if to train the model or try to load a pretrained version.",
    # )

    parser.add_argument(
        "--train_epoch", type=int, default=30,
        help="The number of epoch to train the based model for, default is 30."
    )

    parser.add_argument(
        "--add_early_stopper", type=bool, 
        action=argparse.BooleanOptionalAction, default=True,
        help="Determines if an early stopper will be added for the model training"
    )

    parser.add_argument(
        "--nas_with_training", type=bool, action=argparse.BooleanOptionalAction, default=True,
        help=("Determines if the nas compression configuration search trains the compressed"
              " model or not.")
    )

    parser.add_argument(
        "--nas_seed", type=int, default=-1,
        help=("The seed for the random nas data generation process, it defaults to -1 in " \
        "which case the current time is used as the seed to ensure randomness.")
    )

    parser.add_argument(
        "--nas_num_data", required=True, type=int,
        help=("The number of compressed model from the reduced search space to generate"
              " training examples from.")
    )

    parser.add_argument(
        "--nas_epoch", type=int, default=2,
        help=("The number of epoch to train the compressed model during nas data generation"
             " if with nas_with_training is set,")
    )

    parser.add_argument(
        "--search_min_ram_ratio", type=float, default=0.25,
        help=("The minimum ram ratio to consider when generating nas data, to reduce the"
          " search space.")
    )

    parser.add_argument(
        "--search_min_flash_ratio",  type=float, default=0.25,
        help=("The minimum flash ratio to consider when generating nas data, to reduce the"
          " search space.")
    )

    # parser.add_argument(
    #     "--estimator_hidden_dim", type=int, nargs="+", default=[128, 128],
    #     help="Set the hidden dimensions for the MLP for the model estimator."
    # )

    # parser.add_argument(
    #     "--estimator_dropout", type=float, default=.5,
    #     help="Set the drop out for the MLP for the model estimatior."
    # )

    # parser.add_argument(
    #     "--estimator_epoch", type=int, default=500,
    #     help="The number of epoch to train the model estimator for."
    # )

    # parser.add_argument(
    #     "--max_metric_difference", type=float, default=1,
    #     help=("The maximum metric difference accuracy a compressed model must attain as predicted by the"
    #          " estimator, to be considered valid in the search space.")
    # )

    # parser.add_argument(
    #     "--population_size", type=int, default=50,
    #     help=("The population size for the evolution algorithm for searching for the "
    #           "best model")
    # )

    # parser.add_argument(
    #     "--num_generations", type=int, default=50,
    #     help=("The number of generations for the evolution algorithm for searching for the "
    #           "best model")
    # )

    # parser.add_argument(
    #     "--search_for_smallest_best", type=bool, default=True, 
    #     action=argparse.BooleanOptionalAction,
    #     help=("Determines if to search for the smallest possible compression configuration"
    #           " without considering any board info, basically the unrestricted search.")
    # )

    # parser.add_argument(
    #     "--search_for_smallest_best_to_fit_board", type=bool, default=True,
    #     action=argparse.BooleanOptionalAction,
    #     help=("Detemines if to search the best model (based on the estimator's prediction"
    #           " that can fit into a specified board.")
    # )

    # parser.add_argument(
    #     "--retrain_epoch", type=int, required=True,
    #     help="The number of epochs to retrain the compressed model."
    # )

    # parser.add_argument(
    #     "--retrain_lr", type=float, default=1e-3,
    #     help="The learning rate to use for retraining the compressed model."
    # )

    # parser.add_argument(
    #     "--retrain_lr_scheduler_patience", type=int, default=2,
    #     help="The number of epochs to train before reducing the learning rate."
    # )

    # parser.add_argument(
    #     "--retrain_lr_scheduler_factor", type=float, default=.1,
    #     help="The factor to reduce the lr when the training plateau"
    # )

    parser.add_argument(
        "--env", required=True, choices=["local", "vertexai"],
        help=("The environment you are running the experiment on, either locally or on"
              " vertexai")
    )

    parser.add_argument(
        "--LOCATION", required=True, choices=["us-central1", "us-east1"],
        help=("The location in which your vertexai storage bucket and training is located.")
    )


    args = parser.parse_args()

    return args


if __name__ == "__main__":

    experiment_args = get_experiment_args()

    if  experiment_args.env == VERTEXAI_ENV:
        import google.cloud.logging
        client = google.cloud.logging.Client()
        client.setup_logging(log_level=logging.INFO)
    else:

        log_file = f"{experiment_args.experiment_name}_{datetime.datetime.now().strftime(format='%d-%H-%M')}.log"
        
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=logging.DEBUG,
            format="%(name)s, %(asctime)s, %(levelname)s, %(funcName)s, %(lineno)d, %(message)s, "
        )

    logger = logging.getLogger(__name__)

    logger.info(f"Runing experiment with info: {experiment_args}")

    save_file(experiment_args, experiment_args_file)

    logger.info(f"Nas with trainig {experiment_args.nas_with_training}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    set_random_seed(experiment_args.seed)


    model_name, dataset_name = experiment_args.experiment_name.split("_")

    # Getting the model
    if model_name == "lenet5":
        from .lenet5 import get_model
    elif model_name == "vgg13":
        from .vgg13 import get_model
    elif model_name == "resnet":
        from .resnet import get_model
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Getting the dataset loader
    if dataset_name == "mnist":
        from .mnist import get_data_loaders, get_metric
    elif dataset_name == "cifar100":
        from .cifar100 import get_data_loaders, get_metric
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get a trained model if it exists

    
    train_loader, test_loader = get_data_loaders()
    baseline_model = get_model()
    baseline_model.to(device=DEVICE)
    metric = get_metric()

    # if not experiment_args.train:
    #     pass
    #     # if os.path.exists(trained_file_name):
    #     #     logger.info(f"Loading baseline from {trained_file_name}...")
    #     #     baseline_model.load_state_dict(torch.load(trained_file_name, map_location=DEVICE))
    #     # else:
    #     #     experiment_args.train = True
    # else:
    early_stopper = EarlyStopper(
        monitor_metric="validation_loss", delta=1e-7, mode="min", patience=4, restore_best_state_dict=True
    )
    criterion_fun = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_fun = optim.AdamW(baseline_model.parameters(), lr=1.e-3, weight_decay=1e-2)
    
    # baseline_model.fit(
    #     train_loader, experiment_args.train_epoch, criterion_fun, optimizer_fun,
    #     validation_dataloader=test_loader, metrics={"metric": metric},
    #     callbacks=[early_stopper],
    #     device=DEVICE
    # )

    save_file({
        "model": baseline_model.state_dict(),
        # "optim": optimizer_fun.state_dict()
        }, baseline_model_file)

    original_metric = baseline_model.fuse().evaluate(test_loader, {'metric': metric}, device=DEVICE)['metric']
    logger.info(f"Baseline Accuracy: {original_metric:.4f}%")

    calibration_data = next(iter(train_loader))[0].to(DEVICE)
    
    logger.info("\n--- Generatng NAS Data ---")
    logger.info("Running NAS Sampling (this may take time)...")

    nas_seed = int(time.time()) if experiment_args.nas_seed == -1 else experiment_args.nas_seed
    
    optimizer_kwargs = {
        "momentum": .9,
        "weight_decay": 5e-4
    }
    lr_scheduler_kwargs = {
        "mode": "min",
        "patience": 1
    }

    start_time = time.time()
    nas_parameters = get_nas_compression_data(
        baseline_model,
        experiment_args.input_shape, test_loader, metric, calibration_data, filter=get_model_filter_for_nas(baseline_model, experiment_args.input_shape), device=DEVICE, 
        num_data=experiment_args.nas_num_data, train=experiment_args.nas_with_training, train_dataloader=train_loader, epochs=experiment_args.nas_epoch, 
        criterion_fun=nn.CrossEntropyLoss(), random_seed=nas_seed,
        optimizer_cls=torch.optim.SGD, optimizer_kwargs=optimizer_kwargs, lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau, lr_scheduler_kwargs=lr_scheduler_kwargs
    )

    nas_time = time.time() - start_time
    save_file({
        "nas_parameters": nas_parameters,
        "num_data": experiment_args.nas_num_data,
        "time": nas_time
    }, f"{nas_param_file}-({experiment_args.nas_num_data})-({nas_seed})-(train-{experiment_args.nas_with_training})")

    logger.info(f"Generated {len(list(nas_parameters.values())[0])} NAS samples.")

    # logger.info("\n--- Training Accuracy Estimator ---")
    # estimator = Estimator(
    #     nas_parameters, baseline_model.get_compression_possible_hyperparameters(),
    #     device=DEVICE, hidden_dim=experiment_args.estimator_hidden_dim, dropout=experiment_args.estimator_dropout
    # )
    # estimator_history = estimator.fit(epochs=experiment_args.estimator_epoch)
    # save_file(estimator_history, estimator_training_history_file)

    # if experiment_args.search_for_smallest_best:
    #     logger.info(f"Searching for smallest config with Metric > {experiment_args.max_metric_difference} ...")
    #     # Note: condition depends on estimator scale. Assuming estimator predicts 0-100 accuracy.
    #     best_configuration, best_result_info = evolutionary_search_compression_config(
    #         baseline_model,
    #         estimator,
    #         experiment_args.input_shape,
    #         calibration_data,
    #         condition=get_search_for_smallest_best_condition(baseline_model, experiment_args.input_shape), 
    #         objective= lambda metric, size, ram, config: size, 
    #         filter=get_model_filter_for_nas(baseline_model, experiment_args.input_shape),
    #         maximize=False, # Minimize Size
    #         verbose=True,
    #         population_size=experiment_args.population_size,
    #         generations=experiment_args.num_generations
    #     )

    #     best_configuration = baseline_model.decode_compression_dict_hyperparameter(best_configuration)
    #     logger.info(f"best configuration: {best_configuration}\nbest_result_info: {best_result_info}")

    #     compressed_model = train_compressed_model(
    #         baseline_model, 
    #         experiment_args.retrain_epoch, 
    #         train_loader, 
    #         test_loader, 
    #         best_configuration, 
    #         metric, 
    #         experiment_args.input_shape, 
    #         experiment_args.retrain_lr,
    #         experiment_args.retrain_lr_scheduler_patience,
    #         experiment_args.retrain_lr_scheduler_factor,
    #         DEVICE
    #     )
    #     logger.info("==============Done Retraining for smallest config with Metric================")
    #     metric_value = compressed_model.evaluate(test_loader, {"metric": metric}, device=DEVICE)
    #     logger.info(f"best configuration {best_configuration}\nbest metric: {metric_value} ")
    #     save_file(best_configuration, best_configuration_smallest_metric_file)

    # if experiment_args.search_for_smallest_best_to_fit_board:

    #     logger.info(f"Searching for smallest config to fit in board with")
    #     best_configuration, best_result_info = evolutionary_search_compression_config(
    #         baseline_model,
    #         estimator,
    #         experiment_args.input_shape,
    #         calibration_data,
    #         condition=get_search_for_smallest_best_condition(baseline_model, experiment_args.input_shape), 
    #         objective= lambda metric, size, ram, config: metric, 
    #         filter=get_model_filter_for_nas(baseline_model, experiment_args.input_shape),
    #         maximize=False, # Minimize Size
    #         verbose=True,
    #         population_size=experiment_args.population_size,
    #         generations=experiment_args.num_generations
    #     )

    #     best_configuration = baseline_model.decode_compression_dict_hyperparameter(best_configuration)
    #     logger.info(f"best configuration: {best_configuration}\nbest_result_info: {best_result_info}")
        
    #     compressed_model = train_compressed_model(
    #         baseline_model, 
    #         experiment_args.retrain_epoch, 
    #         train_loader, 
    #         test_loader, 
    #         best_configuration, 
    #         metric, 
    #         experiment_args.input_shape, 
    #         experiment_args.retrain_lr,
    #         experiment_args.retrain_lr_scheduler_patience,
    #         experiment_args.retrain_lr_scheduler_factor,
    #         DEVICE
    #     )

    #     logger.info("==============Done Retraining for smallest config to fit in board================")
    #     metric_value = compressed_model.evaluate(test_loader, {"metric": metric}, device=DEVICE)
    #     logger.info(f"best configuration {best_configuration}\nbest metric: {metric_value} ")

    #     torch.save(best_configuration, best_configuration_smallest_for_board_file)
