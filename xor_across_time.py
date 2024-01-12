from probes import LRProbe
from utils import DataManager
import torch as th
import argparse
from probes import LRProbe
from utils import DataManager
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
import time
from transformers import AutoConfig
from generate_acts import generate_acts

label_names = [
    "has_alice",
    "has_not",
    "label",
    "has_alice xor has_not",
    "has_alice xor label",
    "has_not xor label",
    "has_alice xor has_not xor label",
]

all_checkpoints = (
    [0] + [2**i for i in range(10)] + [1000 * 2**i for i in range(8)] + [143_000]
)


def xor_results(model, device, layers=None, checkpoints=None, compute_acts=True):
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
        print(f"Using device {device}")
    if checkpoints is None:
        checkpoints = all_checkpoints
    if layers is None:
        config = AutoConfig.from_pretrained(model)
        layers = list(range(config.num_hidden_layers + 1))
    if compute_acts:
        for checkpoint in checkpoints:
            print(f"Generating activations for checkpoint {checkpoint}")
            generate_acts(
                model,
                layers,
                ["cities_alice", "neg_cities_alice"],
                device=device,
                revision=f"step{checkpoint}",
            )
    layer_accs = {}
    for layer in layers:
        checkpoint_accs = {}
        for i in checkpoints:
            revision = f"step{i}"
            print(f"Layer {layer}, checkpoint {revision}")
            accs = {}
            for label_name in label_names:
                dm = DataManager()
                for dataset in ["cities_alice", "neg_cities_alice"]:
                    dm.add_dataset(
                        dataset,
                        model,
                        layer,
                        label=label_name,
                        center=False,
                        split=0.8,
                        device=device,
                        revision=revision,
                    )
                acts, labels = dm.get("train")
                probe = LRProbe.from_data(acts, labels, bias=True, device=device)
                acts, labels = dm.get("val")
                acc = (probe(acts).round() == labels).float().mean()
                accs[label_name] = acc.item()
            checkpoint_accs[revision] = accs
        layer_accs[layer] = checkpoint_accs
    return layer_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XOR Across Time")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument(
        "--device", type=str, help="Device (auto, cuda, cpu)", default="auto"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Layers to probe. Default: all",
        default=None,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        help="Checkpoints to probe. Default: All checkpoints on a log scale",
        default=None,
    )
    parser.add_argument(
        "--disable-acts-computation",
        action="store_false",
        default=True,
        help="Set flag to disable computation of activations. If it is not given,"
        "activations will be computed for all checkpoints before probing.",
    )
    args = parser.parse_args()

    all_accs = xor_results(
        args.model,
        args.device,
        layers=args.layers,
        checkpoints=args.checkpoints,
        compute_acts=args.compute_acts,
    )
    df = pd.DataFrame.from_dict(
        {
            (layer, revision): all_accs[layer][revision]
            for layer in all_accs
            for revision in all_accs[layer]
        },
        orient="index",
    )
    df.index.names = ["layer", "revision"]
    df.columns.name = "dataset"
    path = Path("results") / args.model / "xor_across_time"
    path.mkdir(parents=True, exist_ok=True)
    time_id = int(time.time())
    df.to_csv(path / f"results_{time_id}.csv")
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    layers = list(all_accs.keys())
    checkpoint_nbs = list(all_accs[layers[0]].keys())
    for checkpoint_nb in checkpoint_nbs:
        for layer in layers:
            values = [all_accs[layer][checkpoint_nb][key] for key in label_names]
            fig.add_trace(
                go.Bar(
                    x=label_names,
                    y=values,
                    name=f"Layer {layer}, {checkpoint_nb}",
                )
            )

    # Add slider
    fig.update_layout(
        barmode="group",
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "visible": True,
                    "prefix": "Layer: ",
                    "suffix": "",
                },
                "pad": {"b": 10, "t": 50},
                "steps": [
                    {
                        "label": str(layer),
                        "method": "update",
                        "args": [{"visible": [layer == l for l in layers]}],
                    }
                    for layer in layers
                ],
            }
        ],
    )
    for data in fig.data:
        data.update(visible=f"Layer {layers[0]}" in data.name)
        data.name = data.name.split(", ")[1]
    fig.update_yaxes(range=[0, 1])
    fig.write_html(path / f"interactive_{time_id}.html")
