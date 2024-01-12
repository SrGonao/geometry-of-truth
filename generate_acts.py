import torch as th
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
import argparse
import pandas as pd
from tqdm import trange
import configparser
from pathlib import Path
from glob import glob

config = configparser.ConfigParser()
config.read("config.ini")
HF_KEY = config["hf_key"]["hf_key"]
ROOT = Path(__file__).parent


def get_path(
    model_name,
    dataset_name,
    layer=None,
    shuffle=False,
    random_init=False,
    noperiod=False,
    revision=None,
    output_dir=None,
):
    """
    Returns the path to the activations for the specified model, dataset, and layer.

    If output_dir is None, uses ROOT / "acts" as the output directory.
    """
    if output_dir is None:
        output_dir = ROOT / "acts"
    model_dir = output_dir / model_name
    if revision is not None:
        model_dir = model_dir / revision
    if shuffle:
        model_dir = model_dir / "shuffle"
    if random_init:
        model_dir = model_dir / "random_init"
    if noperiod:
        model_dir = model_dir / "noperiod"
    dataset_dir = model_dir / dataset_name
    if layer is None:
        return dataset_dir
    else:
        return dataset_dir / f"layer_{layer}"


def load_model(model_name, device, revision=None, shuffle=False, random_init=False):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_KEY,
        revision=revision,
    )
    if random_init:
        config = AutoConfig.from_pretrained(model_name, revision=revision, token=HF_KEY)
        model = AutoModelForCausalLM.from_config(config, token=HF_KEY)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_KEY,
            revision=revision,
        )
    if shuffle:
        # create reset network by permuting the weights for each parameter
        for param in model.parameters():
            param.data = param.data[..., th.randperm(param.size(-1))]
    # if device != "cpu":
    #     model = model.half()
    model.to(device)
    return tokenizer, model


def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset["statement"].tolist()
    return statements


def get_acts(statements, tokenizer, model, layers, device):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # get last token activations
    acts = {layer: [] for layer in layers}
    for statement in statements:
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        h_states = model(input_ids, output_hidden_states=True).hidden_states
        for layer in layers:
            acts[layer].append(h_states[layer][0, -1].squeeze())

    for layer, act in acts.items():
        acts[layer] = th.stack(act).float()
    return acts


@th.no_grad()
def generate_acts(
    model_name,
    layers,
    datasets,
    output_dir=None,
    noperiod=False,
    device="cpu",
    shuffle=False,
    random_init=False,
    revision=None,
):
    assert (not shuffle) or (not random_init), "Can't shuffle and random init"
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
        print(f"Using device {device}")
    tokenizer, model = load_model(
        model_name, device, revision=revision, shuffle=shuffle, random_init=random_init
    )
    for dataset in datasets:
        statements = load_statements(dataset)
        if noperiod:
            statements = [statement[:-1] for statement in statements]
        layers = (
            layers
            if layers != [-1]
            else list(range(model.config.num_hidden_layers + 1))
        )
        save_dir = get_path(
            model_name,
            dataset,
            noperiod=noperiod,
            shuffle=shuffle,
            random_init=random_init,
            revision=revision,
            output_dir=output_dir,
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        for idx in trange(
            0, len(statements), 25, desc=f"Generating activations for {dataset}"
        ):
            acts = get_acts(
                statements[idx : idx + 25],
                tokenizer,
                model,
                layers,
                device,
            )
            for layer, act in acts.items():
                layer_dir = save_dir / f"layer_{layer}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                th.save(act, layer_dir / f"batch_{idx}.pt")


def collect_acts(
    dataset_name,
    model_name,
    layer,
    center=True,
    scale=False,
    device="cpu",
    noperiod=False,
    shuffle=False,
    random_init=False,
    revision=None,
):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = get_path(
        model_name,
        dataset_name,
        layer=layer,
        noperiod=noperiod,
        shuffle=shuffle,
        random_init=random_init,
        revision=revision,
    )
    if (
        not directory.exists()
        or not any(directory.iterdir())
        or len(glob(str(directory / "batch_*.pt"))) == 0
    ):
        generate_acts(
            model_name,
            [layer],
            [dataset_name],
            ROOT / "acts",
            noperiod=noperiod,
            shuffle=shuffle,
            random_init=random_init,
            device=device,
            revision=revision,
        )
    activation_files = sorted(
        glob(str(directory / "batch_*.pt")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    acts = [th.load(file).to(device) for file in activation_files]
    acts = th.cat(acts, dim=0).to(device)
    if center:
        acts = acts - th.mean(acts, dim=0)
    if scale:
        acts = acts / th.std(acts, dim=0)
    return acts


if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(
        description="Generate activations for statements in a dataset"
    )
    parser.add_argument(
        "--model",
        default="llama-13b",
        help="Size of the model to use. Options are 7B or 30B",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        help="Layers to save embeddings from. Layer 0 correspond to words embeddings. Default is -1, which saves all layers.",
        type=int,
        default=[-1],
    )
    parser.add_argument(
        "--datasets", nargs="+", help="Names of datasets, without .csv extension"
    )
    parser.add_argument(
        "--output_dir", default="acts", help="Directory to save activations to"
    )
    parser.add_argument(
        "--noperiod",
        action="store_true",
        default=False,
        help="Set flag if you don't want to add a period to the end of each statement",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on. Set to 'auto' to use cuda if available",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Revision of model to use. Useful for pythia checkpoints",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Set flag to shuffle weights of model",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        default=False,
        help="Set flag to initialize model with random weights",
    )
    args = parser.parse_args()
    generate_acts(
        args.model,
        args.layers,
        args.datasets,
        output_dir=args.output_dir,
        noperiod=args.noperiod,
        device=args.device,
        shuffle=args.shuffle,
        random_init=args.random_init,
        revision=args.revision,
    )
