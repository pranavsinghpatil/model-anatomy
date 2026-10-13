import io
import json
import logging

from model_anatomy.service.diagnostics import configure_logging, log_event


def captured_logger(level=logging.INFO):
    logger = logging.Logger("model_anatomy.test")  # noqa: LOG001 - isolate tests from application loggers
    logger.setLevel(level)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream, handler


def test_log_event_emits_json_with_explicit_safe_context():
    logger, stream, handler = captured_logger()
    try:
        log_event(
            logger,
            "analysis.failed",
            analysis_id="analysis-123",
            request_id="request-456",
            model_id="org/model",
            revision="a" * 40,
            engine_version="1.2.3",
            stage="construction",
            status="failed",
            code="invalid_config",
            method="POST",
            duration_ms=12.3456,
            cached=False,
        )

        record = json.loads(stream.getvalue())
        assert record["event"] == "analysis.failed"
        assert record["analysis_id"] == "analysis-123"
        assert record["request_id"] == "request-456"
        assert record["model_id"] == "org/model"
        assert record["revision"] == "a" * 40
        assert record["engine_version"] == "1.2.3"
        assert record["stage"] == "construction"
        assert record["status"] == "failed"
        assert record["code"] == "invalid_config"
        assert record["method"] == "POST"
        assert record["duration_ms"] == 12.346
        assert record["cached"] is False
        assert "time" in record and "level" in record
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_exception_log_omits_message_and_locals_but_keeps_type_and_locations():
    logger, stream, handler = captured_logger()
    exception_secret = "exception-secret-7b2d"
    local_secret = "local-secret-c91f"

    def fail_with_private_context():
        private_value = local_secret
        raise RuntimeError(f"provider leaked {exception_secret} {private_value}")

    try:
        try:
            fail_with_private_context()
        except RuntimeError as error:
            log_event(logger, "analysis.failed", exception=error)

        raw = stream.getvalue()
        record = json.loads(raw)
        assert exception_secret not in raw
        assert local_secret not in raw
        assert "provider leaked" not in raw
        assert record["exception_type"] == "RuntimeError"
        assert any(frame["function"] == "fail_with_private_context" for frame in record["frames"])
        assert all(set(frame) == {"file", "function", "line"} for frame in record["frames"])
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_exception_frame_locations_are_bounded_to_twelve():
    logger, stream, handler = captured_logger()

    def fail_deeply(remaining):
        if remaining:
            return fail_deeply(remaining - 1)
        raise LookupError("frame bound")

    try:
        try:
            fail_deeply(20)
        except LookupError as error:
            log_event(logger, "analysis.failed", exception=error)

        record = json.loads(stream.getvalue())
        assert len(record["frames"]) == 12
        assert all(isinstance(frame["line"], int) for frame in record["frames"])
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_trusted_metadata_newlines_are_json_escaped_as_one_record():
    logger, stream, handler = captured_logger()
    event = 'analysis.failed\n"forged-event"'
    model_id = "org/model\nforged-field=true"
    try:
        log_event(logger, event, model_id=model_id)

        output = stream.getvalue()
        assert output.count("\n") == 1
        record = json.loads(output)
        assert record["event"] == event
        assert record["model_id"] == model_id
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_disabled_log_level_emits_nothing():
    logger, stream, handler = captured_logger(logging.WARNING)
    try:
        log_event(logger, "analysis.debug", level=logging.INFO)
        assert stream.getvalue() == ""
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_configure_logging_is_idempotent_and_only_changes_its_namespace():
    logger = logging.getLogger("model_anatomy")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    unrelated = logging.getLogger("third_party.diagnostics_test")
    unrelated_handlers = list(unrelated.handlers)
    unrelated_level = unrelated.level
    unrelated_propagate = unrelated.propagate
    unrelated.addHandler(logging.NullHandler())
    sentinel = unrelated.handlers[-1]
    unrelated.setLevel(logging.ERROR)
    unrelated.propagate = True
    expected_unrelated = (list(unrelated.handlers), unrelated.level, unrelated.propagate)
    stream = io.StringIO()
    capture_handler = logging.StreamHandler(stream)
    capture_handler.setFormatter(logging.Formatter("%(message)s"))

    try:
        configure_logging("INFO")
        configured_count = sum(
            bool(getattr(handler, "anatomy_diagnostics", False)) for handler in logger.handlers
        )
        configure_logging("DEBUG")
        assert sum(bool(getattr(handler, "anatomy_diagnostics", False)) for handler in logger.handlers) == (
            configured_count
        )
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert (list(unrelated.handlers), unrelated.level, unrelated.propagate) == expected_unrelated

        logger.addHandler(capture_handler)
        log_event(logger, "configured.once")
        assert len(stream.getvalue().splitlines()) == 1
    finally:
        logger.removeHandler(capture_handler)
        capture_handler.close()
        for handler in list(logger.handlers):
            if handler not in original_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.handlers[:] = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        unrelated.removeHandler(sentinel)
        unrelated.handlers[:] = unrelated_handlers
        unrelated.setLevel(unrelated_level)
        unrelated.propagate = unrelated_propagate
