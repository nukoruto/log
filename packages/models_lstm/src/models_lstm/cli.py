"""Command line interface for the models-lstm package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .train import run_train_command


def _add_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train an LSTM-based anomaly detector on contract CSV data.",
    )
    parser.add_argument(
        "--normal", type=Path, required=True, help="Path to training contract CSV."
    )
    parser.add_argument(
        "--val", type=Path, required=True, help="Path to validation contract CSV."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for deterministic training.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (<=50).",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=64, help="Embedding dimension for op_category."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension of the LSTM."
    )
    parser.add_argument(
        "--layers", type=int, default=1, help="Number of stacked LSTM layers."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability between LSTM layers.",
    )
    parser.add_argument(
        "--lambda-huber",
        type=float,
        default=1.0,
        help="Weight applied to the Huber regression loss.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Delta parameter for the Huber loss transition.",
    )
    parser.set_defaults(func=_handle_train)


def _handle_train(args: argparse.Namespace) -> int:
    run_train_command(
        normal=args.normal,
        val=args.val,
        out=args.out,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        lambda_huber=args.lambda_huber,
        huber_delta=args.huber_delta,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="models-lstm", description="LSTM anomaly detection utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
