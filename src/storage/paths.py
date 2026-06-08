from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunPaths:
    root: Path
    _pipeline_dir: Path | None = None

    @property
    def pipeline_dir(self) -> Path | None:
        return self._pipeline_dir

    @pipeline_dir.setter
    def pipeline_dir(self, value: Path | str) -> None:
        value = self.root / Path(value)
        if self._pipeline_dir != value:
            value.mkdir(parents=True, exist_ok=True)
        self._pipeline_dir = value

    @property
    def output_dir(self) -> Path:
        if self.pipeline_dir is None:
            return self.root

        return self.pipeline_dir

    @property
    def latest_txt_path(self) -> Path:
        return self.root / "latest.txt"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model"

    @property
    def model_dir(self) -> Path:
        return self.model_path

    @property
    def scores_path(self) -> Path:
        return self.output_dir / "scores.json"

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / "predictions.json"

    @property
    def best_checkpoints_comparison_path(self) -> Path:
        return self.output_dir / "best_checkpoints_comparison.json"

    @property
    def copy_analysis_path(self) -> Path:
        return self.output_dir / "copy_analysis.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    def get_run_dir(self, run_id: int) -> Path:
        return self.root / f"run_{run_id:03d}"

    def checkpoint_dir(self, checkpoint_name: str) -> Path:
        return self.model_path / checkpoint_name

    def checkpoint_predictions_path(self, checkpoint_name: str) -> Path:
        return self.checkpoint_dir(checkpoint_name) / "predictions.json"

    @classmethod
    def for_runs_root(cls, root: Path | None = None) -> RunPaths:
        """Create a RunPaths targeting the repository runs root.

        If `root` is omitted, this defaults to the conventional "runs" folder
        in the current working directory. Callers should use this factory
        instead of embedding Path("runs") literals.
        """
        return cls(root or Path("runs"))
