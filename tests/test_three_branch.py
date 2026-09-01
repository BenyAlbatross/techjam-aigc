from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from techjam_aigc.three_branch.config import ThreeBranchConfig
from techjam_aigc.three_branch.data import (
    ThreeBranchDataset,
    _global_view,
    _native_crops,
)
from techjam_aigc.three_branch.memory import DualPrototypeMemory
from techjam_aigc.three_branch.model import ThreeBranchDetector, score_distribution_pool
from techjam_aigc.three_branch.training import train_ten_epochs


class DummyEncoder(nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.output_dim)
        self.lora_A = nn.Parameter(torch.randn(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pooled = pixels.unfold(2, 8, 8).unfold(3, 8, 8).mean(dim=(-1, -2))
        tokens = pooled.permute(0, 2, 3, 1).flatten(1, 2)
        return self.projection(tokens)


def _memory() -> DualPrototypeMemory:
    generator = torch.Generator().manual_seed(4)
    return DualPrototypeMemory(
        torch.randn(6, 8, generator=generator),
        torch.randn(6, 8, generator=generator),
        topk=3,
        temperature=0.1,
    )


def test_config_enforces_name_all_train_and_ten_epochs() -> None:
    config = ThreeBranchConfig()
    config.validate()
    values = config.to_dict()
    values["optimizer"]["epochs"] = 9
    with pytest.raises(ValueError, match="exactly 10 epochs"):
        ThreeBranchConfig.from_dict(values)
    values = config.to_dict()
    values["data"]["split"] = "val"
    with pytest.raises(ValueError, match="only the train split"):
        ThreeBranchConfig.from_dict(values)


def test_dual_memory_retains_class_relative_direction_and_metrics() -> None:
    memory = _memory()
    tokens = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    output = memory(tokens)
    assert output.authentic.residual.shape == (2, 4, 8)
    assert output.synthetic.residual.shape == (2, 4, 8)
    assert output.authentic.indices.shape == (2, 4, 3)
    assert torch.all(output.authentic.distance >= 0)
    assert torch.all((output.synthetic.entropy >= 0) & (output.synthetic.entropy <= 1.00001))
    assert not torch.allclose(output.authentic.residual, output.synthetic.residual)


def test_three_branch_forward_exposes_each_expert_and_trains_heads() -> None:
    config = ThreeBranchConfig().head
    model = ThreeBranchDetector(DummyEncoder(), _memory(), config)
    model.configure_for_training()
    output = model(
        torch.randn(2, 3, 16, 16),
        torch.rand(2, 5, 3, 32, 32),
    )
    assert output.logit.shape == (2,)
    assert output.global_logit.shape == (2,)
    assert output.memory_logit.shape == (2,)
    assert output.forensic_logit.shape == (2,)
    assert output.patch_logits.shape == (2, 4)
    assert output.crop_logits.shape == (2, 5)
    output.logit.sum().backward()
    assert model.global_head[1].weight.grad is not None
    assert model.memory_patch_head[1].weight.grad is not None
    assert model.forensic_encoder.network[0][0].weight.grad is not None
    assert model.fusion[-1].weight.grad is not None
    assert model.memory.authentic_prototypes.grad is None


def test_score_pool_preserves_mean_dispersion_and_tail() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0, 9.0]])
    pooled = score_distribution_pool(scores, 0.25)
    assert pooled.shape == (1, 4)
    assert pooled[0, 0].item() == 3.0
    assert pooled[0, 2].item() == 9.0
    assert pooled[0, 3].item() == 9.0


def test_native_crops_never_resize_source_pixels() -> None:
    config = ThreeBranchConfig().preprocessing
    grid = np.zeros((240, 300, 3), dtype=np.uint8)
    grid[..., 0] = np.arange(300, dtype=np.uint8)[None, :]
    image = Image.fromarray(grid)
    crops = _native_crops(image, config)
    assert crops.shape == (5, 3, 192, 192)
    assert crops[0, 0, 0, 0] == pytest.approx(grid[0, 0, 0] / 255)
    assert crops[1, 0, 0, -1] == pytest.approx(grid[0, -1, 0] / 255)


def test_global_view_and_dataset_shapes(tmp_path: Path) -> None:
    config = ThreeBranchConfig().preprocessing
    source = tmp_path / "source.png"
    Image.new("RGB", (300, 240), (20, 40, 60)).save(source)
    global_view = _global_view(Image.open(source), config)
    assert global_view.shape == (3, 224, 224)
    frame = pd.DataFrame([{
        "split": "train",
        "source_path": str(source),
        "target": 0,
        "sample_weight": 1.0,
        "group_weight": 1.0,
        "parent_id": "p0",
        "generator_family": "authentic",
        "authentic_subtype": "camera",
        "balance_group": "camera",
        "training_pool": "detector",
    }])
    item = ThreeBranchDataset(frame, config)[0]
    assert item["global_pixels"].shape == (3, 224, 224)
    assert item["native_crops"].shape == (5, 3, 192, 192)


def test_exact_coverage_trainer_writes_final_epoch_ten(tmp_path: Path) -> None:
    base = ThreeBranchConfig()
    config = replace(
        base,
        optimizer=replace(base.optimizer, batch_size=2, workers=0, mixed_precision="fp32"),
    )
    model = ThreeBranchDetector(DummyEncoder(), _memory(), config.head)
    model.configure_for_training()
    rows = [
        {
            "global_pixels": torch.randn(3, 16, 16),
            "native_crops": torch.rand(1, 3, 32, 32),
            "target": index % 2,
            "sample_weight": 1.0,
            "parent_id": f"p{index}",
        }
        for index in range(4)
    ]
    loader = DataLoader(rows, batch_size=2, shuffle=True)
    memory_path = tmp_path / "memory.pt"
    torch.save({"memory": True}, memory_path)
    history = train_ten_epochs(
        model,
        loader,
        config=config,
        device=torch.device("cpu"),
        expected_parent_ids={"p0", "p1", "p2", "p3"},
        provenance={"manifest_sha256": "m", "labels_sha256": "l"},
        memory_path=memory_path,
        output_directory=tmp_path / "run",
    )
    assert len(history) == 10
    assert all(item["rows"] == 4 for item in history)
    final = torch.load(tmp_path / "run" / "three-branch-final.pt", weights_only=True)
    assert final["epoch"] == 10
    assert final["selection"] == {
        "method": "final_epoch",
        "epoch": 10,
        "validation_used": False,
        "generator_holdout_used": False,
    }
