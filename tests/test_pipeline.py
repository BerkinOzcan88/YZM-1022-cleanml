import pandas as pd
import pytest

from cleanml import (
    ConstantStrategy,
    EventHistory,
    MeanStrategy,
    MissingValueImputer,
    Pipeline,
)


def test_pipeline_runs_steps_in_order_and_does_not_mutate_input():
    data = pd.DataFrame({"age": [10.0, None], "city": ["Ankara", None]})
    original = data.copy(deep=True)
    pipeline = Pipeline([
        MissingValueImputer(MeanStrategy(), columns=["age"]),
        MissingValueImputer(ConstantStrategy("Unknown"), columns=["city"]),
    ])

    result = pipeline.fit_transform(data)

    assert result.to_dict("list") == {
        "age": [10.0, 10.0],
        "city": ["Ankara", "Unknown"],
    }
    pd.testing.assert_frame_equal(data, original)


def test_pipeline_observer_records_events_for_successful_fit_transform():
    data = pd.DataFrame({"age": [10.0, None]})
    history = EventHistory()
    pipeline = Pipeline([MissingValueImputer(MeanStrategy(), columns=["age"])])
    pipeline.add_observer(history)

    pipeline.fit_transform(data)

    event_types = [event_type for event_type, _ in history.get_events()]
    assert event_types == [
        "pipeline_started",
        "step_started",
        "step_finished",
        "pipeline_finished",
    ]


def test_pipeline_len_and_repr_show_steps():
    pipeline = Pipeline([
        MissingValueImputer(MeanStrategy(), columns=["age"]),
        MissingValueImputer(ConstantStrategy("Unknown"), columns=["city"]),
    ])

    assert len(pipeline) == 2
    assert repr(pipeline) == "Pipeline(MissingValueImputer -> MissingValueImputer)"


def test_pipeline_rejects_empty_steps():
    with pytest.raises(ValueError, match="at least one step"):
        Pipeline([])


def test_pipeline_rejects_invalid_steps():
    with pytest.raises(TypeError, match="Every step must be a transformer"):
        Pipeline([object()])


def test_pipeline_transform_before_fit_raises_runtime_error():
    pipeline = Pipeline([MissingValueImputer(MeanStrategy(), columns=["age"])])

    with pytest.raises(RuntimeError, match="must be fitted"):
        pipeline.transform(pd.DataFrame({"age": [10.0]}))


def test_pipeline_fit_transform_preserves_original_exception():
    pipeline = Pipeline([MissingValueImputer(MeanStrategy(), columns=["missing"])])

    with pytest.raises(ValueError, match="Columns not found"):
        pipeline.fit_transform(pd.DataFrame({"age": [10.0]}))
