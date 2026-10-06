import pytest
from posthoc.adapt import parse_args


def test_paper_adaptation_defaults_and_trainer_arguments():
    args, training = parse_args(
        [
            "--language",
            "russian",
            "--output_dir",
            "outputs/example",
            "--do_train",
            "--per_device_train_batch_size",
            "4",
        ]
    )
    assert args.language == "russian"
    assert args.train_max_samples == args.train_max_samples_input_distill == 20000
    assert args.k_last_layers == 8
    assert args.lora_r == args.lora_alpha == 256
    assert args.remove_patchscopes_mistakes
    assert not args.include_derivations
    assert not args.preinit_output_embeddings
    assert not args.different_data_per_stage
    assert training.output_dir == "outputs/example"
    assert training.do_train
    assert training.num_train_epochs == 1
    assert training.warmup_ratio == 0.03
    assert training.weight_decay == 0
    assert training.per_device_train_batch_size == 4


def test_unknown_adaptation_option_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--misspelled_option"])
