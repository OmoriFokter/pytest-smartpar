"""Six file conflicts in <project>/output: W-R, W-R, R-W, R-W, W-W, W-W.

Names are chosen so that the conflict kind (seen from the alphabetically first
test) is the one listed. No pair depends on the order: readers accept a missing
file; they only conflict when running *concurrently*.
"""

from ctrl import storage

REPORT = "report:" + "x" * 200
INDEX = '{"items": [' + ", ".join(str(i) for i in range(50)) + "]}"


def _complete_or_missing(content, *expected):
    assert content is None or content in expected


def test_c1_a_write_report():
    storage.write_text("report.txt", REPORT)


def test_c1_b_read_report():
    _complete_or_missing(storage.read_text("report.txt"), REPORT)


def test_c2_a_write_index():
    storage.write_text("index.json", INDEX)


def test_c2_b_read_index():
    _complete_or_missing(storage.read_text("index.json"), INDEX)


def test_c3_a_read_cache():
    _complete_or_missing(storage.read_text("cache.bin"), "cache" * 40)


def test_c3_b_write_cache():
    storage.write_text("cache.bin", "cache" * 40)


def test_c4_a_read_log():
    content = storage.read_text("app.log")
    assert content is None or all(line == "event" for line in content.splitlines())


def test_c4_b_write_log():
    storage.write_text("app.log", "event\n", append=True)


def test_c5_a_write_lock():
    storage.write_text("lock.txt", "a" * 100)
    assert storage.read_text("lock.txt") == "a" * 100


def test_c5_b_write_lock():
    storage.write_text("lock.txt", "b" * 100)
    assert storage.read_text("lock.txt") == "b" * 100


def test_c6_a_export():
    storage.write_text("export.csv", "id,value\n1,a\n" * 10)
    assert storage.read_text("export.csv") == "id,value\n1,a\n" * 10


def test_c6_b_export():
    storage.write_text("export.csv", "id,value\n2,b\n" * 10)
    assert storage.read_text("export.csv") == "id,value\n2,b\n" * 10
