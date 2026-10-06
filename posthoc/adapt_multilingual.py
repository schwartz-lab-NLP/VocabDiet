"""Multilingual entry point using the shared two-stage adaptation pipeline."""

from .adapt import main, parse_args

if __name__ == "__main__":
    args, training_args = parse_args()
    if args.language == "english":
        raise SystemExit(
            "Choose --language spanish, german, russian, or arabic and the corresponding --model_name."
        )
    main(args, training_args)
