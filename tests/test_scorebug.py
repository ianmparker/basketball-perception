from analytics.scorebug import parse_scorebug


def test_parse_standard_scorebug():
    read = parse_scorebug(["8:32", "1ST", "MIN", "9", "NYK", "4"])
    assert read.clock_s == 512
    assert read.period == 1
    assert read.confidence == 1.0


def test_parse_ocr_variants():
    assert parse_scorebug(["8+32", "2ND"]).clock_s == 512
    assert parse_scorebug(["12:00", "4TH"]).clock_s == 720
    assert parse_scorebug(["OT", "2:03"]).period == 5
    assert parse_scorebug(["2:03", "OT"]).clock_s == 123


def test_parse_rejects_invalid_seconds_and_garbage():
    assert parse_scorebug(["8:75", "1ST"]).clock_s is None
    read = parse_scorebug(["MADISON SQUARE GARDEN", "MSG"])
    assert read.clock_s is None
    assert read.period is None
