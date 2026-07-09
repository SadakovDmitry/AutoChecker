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
from .quota_yesno import (
    run_hybrid_router_experiment,
    run_guarded_bayes_yesno_experiment,
    run_quota_yesno_experiment,
    write_guarded_estimate_report,
    write_hybrid_router_report,
    write_quota_yesno_report,
)
from .reports import write_evaluation, write_table
from .subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--text-column",
        default="chat_text",
        help="Column with full chat text. Default: chat_text.",
    )


def _add_subreason_args(parser: argparse.ArgumentParser, *, allow_grouping: bool = True) -> None:
    parser.add_argument(
        "--subreason-map",
        default=None,
        help=(
            "Optional YAML mapping from dataset+iteration+reason_id to stable subreason_key. "
            "Use it when reason numbers changed between prompt iterations."
        ),
    )
    if allow_grouping:
        parser.add_argument(
            "--group-by-subreason-key",
            action="store_true",
            help=(
                "After applying --subreason-map, train/evaluate/verify by stable subreason_key "
                "instead of raw reason_id."
            ),
        )


def _apply_subreason_options(
    frame: pd.DataFrame,
    *,
    mapping_path: str | None,
    group_by_subreason_key: bool = False,
) -> pd.DataFrame:
    if "subreason_key" in frame.columns:
        return use_subreason_key_as_reason_id(frame) if group_by_subreason_key else frame
    mapping = load_subreason_mapping(mapping_path)
    if mapping is None:
        return frame
    mapped = apply_subreason_mapping(frame, mapping)
    if group_by_subreason_key:
        mapped = use_subreason_key_as_reason_id(mapped)
    return mapped


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
        "--target-no-precision",
        type=float,
        default=0.97,
        help="Target precision for automatic 'нет' decisions when --enable-auto-no is used.",
    )
    train.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    train.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformers.")
    train.add_argument(
        "--enable-auto-no",
        action="store_true",
        help="Enable automatic 'нет' decisions. Default is safer auto_yes/review mode.",
    )
    train.add_argument(
        "--max-auto-no-p-correct",
        type=float,
        default=None,
        help=(
            "Optional extra safety cap for automatic 'нет': accept auto_no only when "
            "p_correct is at or below this value, for example 0.04."
        ),
    )
    train.add_argument("--rules", default=None, help="Optional YAML rules file.")
    train.add_argument("--min-reason-samples", type=int, default=8)
    train.add_argument("--min-class-samples", type=int, default=2)
    _add_subreason_args(train)

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
    _add_subreason_args(prepare, allow_grouping=False)

    verify = subparsers.add_parser("verify", help="Verify new classifier results.")
    verify.add_argument("--model", required=True, help="Model directory.")
    verify.add_argument("--input", required=True, help="Input Excel/CSV/JSONL file.")
    verify.add_argument("--output", required=True, help="Output Excel/CSV file.")
    _add_common_data_args(verify)
    _add_subreason_args(verify)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate auto-accept quality on labeled rows.")
    evaluate.add_argument("--model", required=True, help="Model directory.")
    evaluate.add_argument("--data", nargs="+", required=True, help="Evaluation files or glob patterns.")
    evaluate.add_argument("--output", required=True, help="Output report path.")
    _add_common_data_args(evaluate)
    _add_subreason_args(evaluate)

    quota_yesno = subparsers.add_parser(
        "evaluate-quota-yesno",
        help=(
            "Experimental mode: train on previous manual checks, then force every "
            "latest row into да/нет using per-reason historical yes-rate quotas."
        ),
    )
    quota_yesno.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Previous manually checked rows used for training and per-reason yes-rate quotas.",
    )
    quota_yesno.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Latest manually checked rows used only for final comparison.",
    )
    quota_yesno.add_argument("--output", required=True, help="Output report path.")
    _add_common_data_args(quota_yesno)
    quota_yesno.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    quota_yesno.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformers.")
    quota_yesno.add_argument("--rules", default=None, help="Optional YAML rules file.")
    quota_yesno.add_argument("--min-reason-samples", type=int, default=8)
    quota_yesno.add_argument("--min-class-samples", type=int, default=2)
    _add_subreason_args(quota_yesno)
    quota_yesno.add_argument(
        "--strategy",
        choices=("threshold", "quota", "latest-prior"),
        default="latest-prior",
        help=(
            "threshold: learn p_correct cutoff per reason on previous labels; "
            "quota: force latest yes-rate to all-history positive-rate quota; "
            "latest-prior: force latest yes-rate to the latest previous iteration positive-rate quota."
        ),
    )

    guarded_yesno = subparsers.add_parser(
        "evaluate-guarded-yesno",
        help=(
            "Experimental full yes/no mode: estimate prompt accuracy by "
            "guarded_bayes_k40 + offset learned on n-1, then mark every row да/нет."
        ),
    )
    guarded_yesno.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Previous manually checked rows. Latest sheet inside this data is used to learn offset.",
    )
    guarded_yesno.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Latest rows to estimate and force into да/нет.",
    )
    guarded_yesno.add_argument("--output", required=True, help="Output report path.")
    _add_common_data_args(guarded_yesno)
    guarded_yesno.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    guarded_yesno.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformers.")
    guarded_yesno.add_argument("--rules", default=None, help="Optional YAML rules file.")
    guarded_yesno.add_argument("--min-reason-samples", type=int, default=8)
    guarded_yesno.add_argument("--min-class-samples", type=int, default=2)
    guarded_yesno.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Override learned offset. Omit to learn offset from n-1.",
    )
    guarded_yesno.add_argument(
        "--min-offset",
        type=float,
        default=0.0,
        help="Lower bound for learned offset. Default 0.0 keeps the correction protective.",
    )
    guarded_yesno.add_argument(
        "--max-offset",
        type=float,
        default=0.15,
        help="Upper bound for learned offset. Default 0.15 avoids over-correction.",
    )
    guarded_yesno.add_argument(
        "--guard-gap",
        type=float,
        default=0.10,
        help="Do not let guarded estimate fall below topic history rate minus this value.",
    )
    guarded_yesno.add_argument(
        "--bayes-k",
        type=float,
        default=40.0,
        help="History strength for guarded Bayes smoothing.",
    )
    _add_subreason_args(guarded_yesno)

    hybrid = subparsers.add_parser(
        "evaluate-hybrid-router",
        help=(
            "Experimental hybrid mode: low-risk subreasons get full yes/no; "
            "high-risk subreasons get safe auto_yes/review."
        ),
    )
    hybrid.add_argument("--train", nargs="+", required=True)
    hybrid.add_argument("--data", nargs="+", required=True)
    hybrid.add_argument("--output", required=True)
    _add_common_data_args(hybrid)
    hybrid.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    hybrid.add_argument("--no-embeddings", action="store_true")
    hybrid.add_argument("--rules", default=None)
    hybrid.add_argument("--target-precision", type=float, default=0.80)
    hybrid.add_argument("--min-reason-samples", type=int, default=8)
    hybrid.add_argument("--min-class-samples", type=int, default=2)
    hybrid.add_argument("--offset", type=float, default=None)
    hybrid.add_argument("--min-offset", type=float, default=0.0)
    hybrid.add_argument("--max-offset", type=float, default=0.15)
    hybrid.add_argument("--guard-gap", type=float, default=0.10)
    hybrid.add_argument("--bayes-k", type=float, default=40.0)
    hybrid.add_argument("--min-history-rows", type=int, default=10)
    hybrid.add_argument("--max-history-std", type=float, default=0.05)
    hybrid.add_argument("--max-model-history-gap", type=float, default=0.35)
    hybrid.add_argument("--min-estimated-accuracy", type=float, default=0.70)
    hybrid.add_argument(
        "--min-full-mean-p-correct",
        type=float,
        default=0.65,
        help=(
            "Minimum current mean p_correct for a subreason required to allow "
            "full yes/no. This prevents strong history from overriding a weak new batch."
        ),
    )
    hybrid.add_argument(
        "--min-full-train-row-accuracy",
        type=float,
        default=0.65,
        help=(
            "Minimum past row-level accuracy of the personal p_correct threshold "
            "required to allow full yes/no for a subreason."
        ),
    )
    hybrid.add_argument(
        "--max-full-train-rate-gap",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed historical positive-rate gap for the personal "
            "p_correct threshold before routing the subreason to safe mode."
        ),
    )
    hybrid.add_argument(
        "--estimate-strategy",
        choices=("max-history-latest", "guarded-bayes"),
        default="max-history-latest",
        help=(
            "How to estimate subreason prompt accuracy for routing. "
            "max-history-latest uses max(mean p_correct, latest subreason history); "
            "guarded-bayes uses the older guarded Bayes estimate."
        ),
    )
    hybrid.add_argument(
        "--full-yesno-strategy",
        choices=("threshold", "legacy_quota"),
        default="threshold",
        help=(
            "How low-risk subreasons are fully labeled. "
            "threshold uses personal p_correct cutoff per subreason. "
            "legacy_quota keeps old top-N behavior only for reproducing old experiments."
        ),
    )
    _add_subreason_args(hybrid)

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    config = ValidatorConfig(
        target_precision=args.target_precision,
        target_no_precision=args.target_no_precision,
        min_reason_samples=args.min_reason_samples,
        min_class_samples=args.min_class_samples,
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embeddings,
        enable_auto_no=args.enable_auto_no,
        max_auto_no_p_correct=args.max_auto_no_p_correct,
        rules=rules,
    )
    frame = load_tables(
        args.data,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    frame = _apply_subreason_options(
        frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    model = HybridValidator.train(frame, config)
    model.save(args.out)
    print(f"Saved model to {args.out}")
    if model.embedding_error:
        print(model.embedding_error)
    print(model.summary_frame().to_string(index=False))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    prepared, stats = prepare_training_data(
        labels_paths=args.labels,
        messages_paths=args.messages,
        output=None if args.subreason_map else args.output,
        require_answer=not args.allow_unlabeled,
        labels_sheet=args.labels_sheet,
        messages_sheet=args.messages_sheet,
    )
    if args.subreason_map:
        prepared = _apply_subreason_options(
            prepared,
            mapping_path=args.subreason_map,
            group_by_subreason_key=False,
        )
        write_table(prepared, args.output)
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
    frame = _apply_subreason_options(
        frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
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
    frame = _apply_subreason_options(
        frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    predictions = model.predict(frame)
    write_evaluation(predictions, args.output)
    print(f"Wrote evaluation report to {args.output}")
    return 0


def cmd_evaluate_quota_yesno(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    config = ValidatorConfig(
        min_reason_samples=args.min_reason_samples,
        min_class_samples=args.min_class_samples,
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embeddings,
        rules=rules,
    )
    train_frame = load_tables(
        args.train,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    evaluation_frame = load_tables(
        args.data,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    train_frame = _apply_subreason_options(
        train_frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    evaluation_frame = _apply_subreason_options(
        evaluation_frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    model = HybridValidator.train(train_frame, config)
    result = run_quota_yesno_experiment(
        train_frame=train_frame,
        evaluation_frame=evaluation_frame,
        model=model,
        strategy=args.strategy,
    )
    write_quota_yesno_report(result, args.output)
    print(f"Wrote quota yes/no report to {args.output}")
    if model.embedding_error:
        print(model.embedding_error)
    overall = result.summary[result.summary["reason_id"] == "__overall_weighted__"]
    if not overall.empty:
        row = overall.iloc[0]
        print(
            "overall: manual_accuracy={:.2%}; auto_estimated_accuracy={:.2%}; "
            "gap={:.2f} pp; row_label_accuracy={:.2%}".format(
                row["manual_prompt_accuracy"],
                row["auto_estimated_prompt_accuracy"],
                row["accuracy_gap_pp"],
                row["row_label_accuracy"],
            )
        )
    return 0


def cmd_evaluate_guarded_yesno(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    config = ValidatorConfig(
        min_reason_samples=args.min_reason_samples,
        min_class_samples=args.min_class_samples,
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embeddings,
        rules=rules,
    )
    train_frame = load_tables(
        args.train,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    evaluation_frame = load_tables(
        args.data,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    train_frame = _apply_subreason_options(
        train_frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    evaluation_frame = _apply_subreason_options(
        evaluation_frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    result = run_guarded_bayes_yesno_experiment(
        train_frame=train_frame,
        evaluation_frame=evaluation_frame,
        config=config,
        offset=args.offset,
        k=args.bayes_k,
        guard_gap=args.guard_gap,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
    )
    write_guarded_estimate_report(result, args.output)
    print(f"Wrote guarded yes/no report to {args.output}")
    offset = result.offset_summary.iloc[0]["offset"] if not result.offset_summary.empty else 0.0
    source = result.offset_summary.iloc[0]["source"] if not result.offset_summary.empty else ""
    print(f"offset={offset:.4f}; source={source}")
    overall = result.summary[result.summary["reason_id"] == "__overall_weighted__"]
    if not overall.empty:
        row = overall.iloc[0]
        row_accuracy = row.get("row_label_accuracy", float("nan"))
        print(
            "overall: manual_accuracy={:.2%}; estimated_accuracy={:.2%}; "
            "gap={:.2f} pp; abs_gap={:.2f} pp; row_label_accuracy={:.2%}".format(
                row["manual_prompt_accuracy"],
                row["estimated_prompt_accuracy"],
                row["accuracy_gap_pp"],
                row["abs_accuracy_gap_pp"],
                row_accuracy,
            )
        )
    return 0


def cmd_evaluate_hybrid_router(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    config = ValidatorConfig(
        target_precision=args.target_precision,
        min_reason_samples=args.min_reason_samples,
        min_class_samples=args.min_class_samples,
        embedding_model=args.embedding_model,
        use_embeddings=not args.no_embeddings,
        rules=rules,
    )
    train_frame = load_tables(
        args.train,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    evaluation_frame = load_tables(
        args.data,
        text_column=args.text_column,
        require_text=True,
        require_answer=True,
    )
    train_frame = _apply_subreason_options(
        train_frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    evaluation_frame = _apply_subreason_options(
        evaluation_frame,
        mapping_path=args.subreason_map,
        group_by_subreason_key=args.group_by_subreason_key,
    )
    result = run_hybrid_router_experiment(
        train_frame=train_frame,
        evaluation_frame=evaluation_frame,
        config=config,
        offset=args.offset,
        k=args.bayes_k,
        guard_gap=args.guard_gap,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        min_history_rows=args.min_history_rows,
        max_history_std=args.max_history_std,
        max_model_history_gap=args.max_model_history_gap,
        min_estimated_accuracy=args.min_estimated_accuracy,
        min_full_mean_p_correct=args.min_full_mean_p_correct,
        min_full_train_row_accuracy=args.min_full_train_row_accuracy,
        max_full_train_rate_gap=args.max_full_train_rate_gap,
        full_yesno_strategy=args.full_yesno_strategy,
        estimate_strategy=args.estimate_strategy.replace("-", "_"),
    )
    write_hybrid_router_report(result, args.output)
    print(f"Wrote hybrid router report to {args.output}")
    overall = result.summary[result.summary["reason_id"] == "__overall_weighted__"]
    if not overall.empty:
        row = overall.iloc[0]
        print(
            "overall: coverage={:.2%}; precision={:.2%}; errors={}; auto_rows={}; review_rows={}".format(
                row["coverage"],
                row["precision"],
                int(row["errors"]),
                int(row["auto_rows"]),
                int(row["review_rows"]),
            )
        )
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
        if args.command == "evaluate-quota-yesno":
            return cmd_evaluate_quota_yesno(args)
        if args.command == "evaluate-guarded-yesno":
            return cmd_evaluate_guarded_yesno(args)
        if args.command == "evaluate-hybrid-router":
            return cmd_evaluate_hybrid_router(args)
    except (DataFormatError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
