import pathlib

from utils import plot_boosted_training_history


def _history(train_loss, train_acc, test_loss, test_acc):
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def test_plot_boosted_training_history_writes_summary_and_round_plots(tmp_path):
    save_path = tmp_path / "boosted.png"
    round_histories = [
        _history([0.8, 0.6], [0.5, 0.7], [0.9, 0.65], [0.45, 0.68]),
        _history([0.55, 0.4], [0.72, 0.82], [0.6, 0.42], [0.7, 0.84]),
    ]

    result = plot_boosted_training_history(round_histories, save_path=str(save_path))

    assert pathlib.Path(result["summary_path"]).exists()
    assert len(result["round_paths"]) == 2
    assert pathlib.Path(result["round_paths"][0]).exists()
    assert pathlib.Path(result["round_paths"][1]).exists()
