from argparse import ArgumentParser
from pathlib import Path

import torch

from main import export_torchscript_model, load_train_config
from train.gru_model import GruModel


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def export_checkpoint_to_torchscript(config_path, checkpoint_path, output_path):
    config = load_train_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GruModel(**config["model"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return export_torchscript_model(
        model=model,
        output_path=output_path,
        sequence_length=config["training"]["sequence_length"],
        device=device,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="models/gru_model.pt")
    parser.add_argument("--output", default="models/gru_model_torchscript.pt")
    args = parser.parse_args()

    output_path = export_checkpoint_to_torchscript(
        config_path=args.config,
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
    )
    print(f"Exported TorchScript model to {output_path}")


if __name__ == "__main__":
    main()
