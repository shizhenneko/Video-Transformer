from utils.counter import APICounter, APILimitExceeded


def test_counter_increments_and_limits():
    counter = APICounter(max_calls=2)
    assert counter.can_call()
    assert counter.increment("gemini")
    assert counter.increment("kimi")
    assert counter.current_count == 2
    assert not counter.can_call()

    try:
        counter.increment("extra")
    except APILimitExceeded:
        pass
    else:
        raise AssertionError("Expected APILimitExceeded")


def test_counter_reset():
    counter = APICounter(max_calls=1, current_count=1)
    counter.reset()
    assert counter.current_count == 0
    assert counter.can_call()