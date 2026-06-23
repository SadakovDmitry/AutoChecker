from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .config import ValidatorConfig, load_rules
from .data import DataFormatError, load_tables
from .model import HybridValidator
from .prepare import prepare_training_data
from .reports import write_evaluation, write_table


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--text-column",
        default="chat_text",
        help="Column with full chat text. Default: chat_text.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_classifier",
        description="Hybrid validator for classifier labels.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train validators from manually checked rows.")
    train.add_argument("--data", nargs="+", required=True, help="Excel/CSV/JSONL files or glob patterns.")
    _add_common_data_args(train)
    train.add_argument("--out", required=True, help="Directory to save trained model.")
    train.add_argument("--target-precision", type=float, default=0.95)
    train.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    train.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformers.")
    train.add_argument("--rules", default=None, help="Optional YAML rules file.")
    train.add_argument("--min-reason-samples", type=int, default=8)
    train.add_argument("--min-class-samples", type=int, default=2)

    prepare = subparsers.add_parser(
        "prepare",
        help="Merge manual labels with one-message-per-row chat exports.",
    )
    prepare.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Manual check files with chat_id/comm_id, reason_id and да/нет.",
    )
    prepare.add_argument(
        "--messages",
        nargs="+",
        required=True,
        help="Message export files with ID_diologa, Vremya, Kto, Soobschenie.",
    )
    prepare.add_argument("--output", required=True, help="Prepared CSV/XLSX output.")
    prepare.add_argument(
        "--labels-sheet",
        default=None,
        help="Use only one sheet from labels workbook, for example: 'итерация 6'.",
    )
    prepare.add_argument(
        "--messages-sheet",
        default=None,
        help="Use only one sheet from messages workbook.",
    )
    prepare.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Do not require да/нет column. Useful for preparing verify input.",
    )

    verify = subparsers.add_parser("verify", help="Verify new classifier results.")
    verify.add_argument("--model", required=True, help="Model directory.")
    verify.add_argument("--input", required=True, help="Input Excel/CSV/JSONL file.")
    verify.add_argument("--output", required=True, help="Output Excel/CSV file.")
    _add_common_data_args(verify)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate auto-accept quality on labeled rows.")
    evaluate.add_argument("--model", required=True, help="Model directory.")
    evaluate.add_argument("--data", nargs="+", required=True, help="Evaluation files or glob patterns.")
    evaluate.add_argument("--output", required=True, help="Output report path.")
    _add_common_data_args(evaluate)

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    config = ValidatorConfig(
        target_precision=args.target_precision,
        min_reason_samples=args.min_reason_samples,
        min_class_samples=args.min_class_samples,
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embeddings,
        rules=rules,
    )
    frame = load_tables(
        args.data,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    model = HybridValidator.train(frame, config)
    model.save(args.out)
    print(f"Saved model to {args.out}")
    if model.embedding_error:
        print(model.embedding_error)
    print(model.summary_frame().to_string(index=False))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    _, stats = prepare_training_data(
        labels_paths=args.labels,
        messages_paths=args.messages,
        output=args.output,
        require_answer=not args.allow_unlabeled,
        labels_sheet=args.labels_sheet,
        messages_sheet=args.messages_sheet,
    )
    print(f"Wrote prepared dataset to {args.output}")
    print(
        "labels_rows={labels_rows}; prepared_rows={prepared_rows}; "
        "message_rows={message_rows}; message_chats={message_chats}; "
        "matched_rows={matched_rows}; missing_text_rows={missing_text_rows}".format(
            **stats.__dict__
        )
    )
    if stats.missing_text_rows:
        print(
            "warning: some label rows did not match message exports by chat_id/comm_id",
            file=sys.stderr,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    model = HybridValidator.load(args.model)
    frame = load_tables(
        [args.input],
        text_column=args.text_column,
        require_text=True,
        require_answer=False,
    )
    predictions = model.predict(frame)
    write_table(predictions, args.output)
    accepted = int((predictions["decision"] == "accept").sum()) if "decision" in predictions else 0
    print(f"Wrote {len(predictions)} rows to {args.output}; accepted={accepted}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = HybridValidator.load(args.model)
    frame = load_tables(
        args.data,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    predictions = model.predict(frame)
    write_evaluation(predictions, args.output)
    print(f"Wrote evaluation report to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"sklearn\.utils\.extmath",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return cmd_train(args)
        if args.command == "prepare":
            return cmd_prepare(args)
        if args.command == "verify":
            return cmd_verify(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
    except (DataFormatError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
