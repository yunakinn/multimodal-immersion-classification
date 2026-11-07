import numpy as np
import pytest
import torch
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
    main(["--data-dir", str(tmp_path), "--split-dir", str(splits), "--split-kind", "subject", "--output-dir", str(output), "--epochs", "1", "--batch-size", "2"])
    assert (output / "best_model.pt").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "learning_curves.png").is_file()
