from cleanml import ConsoleLogger, EventHistory


def test_event_history_stores_and_clears_events():
    history = EventHistory()

    history.on_event("step_started", {"step_name": "ExampleStep"})

    assert history.get_events() == [("step_started", {"step_name": "ExampleStep"})]

    history.clear()

    assert history.get_events() == []


def test_console_logger_prints_known_events(capsys):
    logger = ConsoleLogger()

    logger.on_event("pipeline_started", {})
    logger.on_event("step_started", {"step_name": "ExampleStep"})
    logger.on_event("step_finished", {"step_name": "ExampleStep"})
    logger.on_event("pipeline_finished", {})
    logger.on_event("error_occurred", {"error": ValueError("bad data")})

    assert capsys.readouterr().out.splitlines() == [
        "[PIPELINE STARTED]",
        "[START] ExampleStep",
        "[DONE] ExampleStep",
        "[PIPELINE FINISHED]",
        "[ERROR] bad data",
    ]
