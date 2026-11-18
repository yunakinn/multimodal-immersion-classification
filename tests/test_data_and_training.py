import numpy as np
import json
import pytest
import torch
from unittest.mock import Mock
import train
from dataset import BiosignalDataset, validate_split_isolation
from model import FusionModel, to_channels_first
from train import main


def fixture_data(root):
    for modality, shape in [("EEG_sliced", (3, 4, 64)), ("GSR_sliced", (3, 64)), ("PPG_sliced", (3, 64))]:
        for label in ["0", "1"]:
            dest = root / modality / label
            dest.mkdir(parents=True)
            for subject in ["S01", "S010", "S02"]:
                np.save(dest / f"{subject}_session.npy", np.zeros(shape, np.float32))
    splits = root / "splits"
    splits.mkdir()
    for name, key in [("train", "S01"), ("val", "S010"), ("test", "S02")]:
        (splits / f"{name}.txt").write_text(key + "\n")
    return splits


def test_subject_ids_do_not_prefix_match(tmp_path):
    splits = fixture_data(tmp_path)
    datasets = [BiosignalDataset(tmp_path, split, splits, verbose=False, split_kind="subject") for split in ["train", "val", "test"]]
    assert len(datasets[0]) == 6
    validate_split_isolation(datasets, check_subjects=True)
    (splits / "test.txt").write_text("S01\n")
    overlap = BiosignalDataset(tmp_path, "test", splits, verbose=False, split_kind="subject")
    with pytest.raises(ValueError, match="overlap"):
        validate_split_isolation([datasets[0], overlap])


def test_exact_file_keys_and_window_cache(tmp_path):
    splits = fixture_data(tmp_path)
    (splits / "train.txt").write_text("S01_session.npy\n")
    data = BiosignalDataset(tmp_path, "train", splits, verbose=False, load_into_memory=True)
    assert len(data) == 6
    assert all(sample[0].base is None for sample in data.samples)
    assert data[0][0].shape == (4, 64)
    mapped = BiosignalDataset(tmp_path, "train", splits, verbose=False)
    assert mapped.samples is None
    assert torch.equal(data[0][0], mapped[0][0])


def test_model_accepts_explicit_channels_first_only():
    with pytest.raises(ValueError, match="batch"):
        to_channels_first(torch.zeros(2, 1, 4, 64))
    model = FusionModel(dropout=0).eval()
    logits, attention = model(torch.zeros(2, 4, 64), torch.zeros(2, 64), torch.zeros(2, 64))
    assert logits.shape == (2, 1)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2, 1))


def test_tiny_training_writes_checkpoint_and_metrics(tmp_path):
    torch.set_num_threads(1)
    splits = fixture_data(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    # A successful run must replace this unusable checkpoint before reading it.
    (output / "best_model.pt").write_bytes(b"previous experiment checkpoint")
    main(["--data-dir", str(tmp_path), "--split-dir", str(splits), "--split-kind", "subject", "--output-dir", str(output), "--epochs", "1", "--batch-size", "2"])
    assert (output / "best_model.pt").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "learning_curves.png").is_file()


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("phase", ["training", "validation"])
def test_failed_epoch_never_loads_previous_checkpoint(tmp_path, monkeypatch, bad_loss, phase):
    splits = fixture_data(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    previous = b"previous experiment checkpoint"
    (output / "best_model.pt").write_bytes(previous)
    train_epoch = Mock(return_value=(bad_loss if phase == "training" else .5, .5))
    eval_epoch = Mock(return_value=(bad_loss if phase == "validation" else .5, .5))
    load = Mock(side_effect=AssertionError("A failed run must not load a checkpoint"))
    predict = Mock(side_effect=AssertionError("A failed run must not evaluate the test split"))
    monkeypatch.setattr(train, "run_train_epoch", train_epoch)
    monkeypatch.setattr(train, "run_eval_epoch", eval_epoch)
    monkeypatch.setattr(train, "load_checkpoint", load)
    monkeypatch.setattr(train, "predict", predict)
    with pytest.raises(ValueError, match=f"Non-finite {phase} loss"):
        main(["--data-dir", str(tmp_path), "--split-dir", str(splits), "--split-kind", "subject",
              "--output-dir", str(output), "--epochs", "1"])
    load.assert_not_called()
    predict.assert_not_called()
    if phase == "training":
        eval_epoch.assert_not_called()
    assert (output / "best_model.pt").read_bytes() == previous
    assert not (output / "metrics.json").exists()


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("phase", ["training", "validation"])
def test_non_finite_batch_loss_is_rejected_before_optimizer_step(bad_loss, phase):
    model = FusionModel(dropout=0)
    batch = (torch.zeros(2, 4, 64), torch.zeros(2, 64), torch.zeros(2, 64), torch.zeros(2))
    optimizer = Mock()

    def criterion(logits, labels):
        return logits.sum() * 0 + bad_loss

    with pytest.raises(ValueError, match=f"Non-finite {phase} loss"):
        if phase == "training":
            train.run_train_epoch(model, [batch], criterion, optimizer, torch.device("cpu"))
        else:
            train.run_eval_epoch(model, [batch], criterion, torch.device("cpu"))
    optimizer.step.assert_not_called()
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("rate", ["nan", "inf", "-inf"])
def test_non_finite_learning_rate_is_rejected_before_data_loading(tmp_path, rate):
    with pytest.raises(ValueError, match="learning rate"):
        main(["--lr=" + rate, "--output-dir", str(tmp_path / "unused")])
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("kind,known,missing", [
    ("file", "S01_session.npy", "missing.npy"),
    ("file", "0/S01_session.npy", "0/missing.npy"),
    ("subject", "S01", "S99"),
])
@pytest.mark.parametrize("extension", ["txt", "csv", "json", "npy"])
def test_unmatched_split_entries_are_not_silently_dropped(tmp_path, kind, known, missing, extension):
    fixture_data(tmp_path)
    split = tmp_path / f"selected.{extension}"
    keys = [known, missing]
    if extension == "json":
        split.write_text(json.dumps({"train": keys}))
    elif extension == "npy":
        np.save(split, np.asarray(keys))
    else:
        split.write_text("\n".join(keys) + "\n")
    with pytest.raises(ValueError, match="Unmatched entries in train split") as error:
        BiosignalDataset(tmp_path, "train", split, split_kind=kind, verbose=False)
    assert missing in str(error.value)


def test_class_relative_split_does_not_require_both_classes(tmp_path):
    splits = fixture_data(tmp_path)
    (splits / "train.txt").write_text("0/S01_session.npy\n")
    selected = BiosignalDataset(tmp_path, "train", splits, verbose=False)
    assert len(selected) == 3
    assert all(item[3] == 0 for item in selected.index)
