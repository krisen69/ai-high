from app.utils.timecode import sec_to_tc, tc_to_sec


def test_timecode_roundtrip():
    tc = sec_to_tc(3723.456)
    assert tc.startswith("01:02:03")
    assert abs(tc_to_sec(tc) - 3723.456) < 0.001
