# Tests deterministic validation/OOD few-shot adaptation curve evaluation.
import numpy as np
import pytest

from hrl.training import evaluate_meta_adaptation


class ZeroMetaWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        # The hierarchical base observation includes service and episode state.
        assert observation.shape == (1639,)
        return np.zeros(18, dtype=np.float32), None


def test_meta_adaptation_curve_reports_pre_and_post_context_metrics() -> None:
    curve = evaluate_meta_adaptation(
        ZeroMetaWorker(),
        split="validation",
        task_ids=(1, 4),
        adaptation_episodes=(0, 1),
        evaluation_episodes=1,
        task_seed=19,
        seed=31,
    )

    assert curve.split == "validation"
    assert [point.adaptation_episodes for point in curve.points] == [0, 1]
    assert curve.points[0].mean_context_transitions == 16.0
    assert curve.points[1].mean_context_transitions == 32.0
    assert all(np.isfinite(point.mean_reward) for point in curve.points)


def test_meta_adaptation_curve_is_deterministic_and_accepts_ood_tasks() -> None:
    arguments = {
        "split": "ood",
        "task_ids": (2,),
        "adaptation_episodes": (0, 2),
        "evaluation_episodes": 1,
        "task_seed": 23,
        "seed": 37,
    }
    first = evaluate_meta_adaptation(ZeroMetaWorker(), **arguments)
    second = evaluate_meta_adaptation(ZeroMetaWorker(), **arguments)

    assert first == second
    assert first.points[-1].mean_context_transitions == 48.0


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("split", {"split": "train"}),
        ("strictly increasing", {"adaptation_episodes": (1, 0)}),
        ("strictly increasing", {"adaptation_episodes": (0, 0)}),
        ("evaluation_episodes", {"evaluation_episodes": 0}),
    ],
)
def test_meta_adaptation_curve_rejects_invalid_evaluation_inputs(
    keyword: str, value: dict[str, object],
) -> None:
    arguments: dict[str, object] = {"split": "validation", "task_ids": (0,)}
    arguments.update(value)
    with pytest.raises(ValueError, match=keyword):
        evaluate_meta_adaptation(ZeroMetaWorker(), **arguments)
