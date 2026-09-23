from polyasr_align import group_chars_into_sentences


def test_english_word_units_keep_forced_alignment_timestamps():
    units = [
        {"text": "Hello", "start": 1.0, "end": 1.5},
        {"text": "world", "start": 1.6, "end": 2.0},
        {"text": "Next", "start": 3.0, "end": 3.4},
        {"text": "thought", "start": 3.5, "end": 4.0},
    ]
    got = group_chars_into_sentences(units, "Hello world. Next thought.")
    assert [x["text"] for x in got] == ["Hello world.", "Next thought."]
    assert [(x["start"], x["end"]) for x in got] == [(1.0, 2.0), (3.0, 4.0)]


def test_character_units_keep_existing_path():
    units = [
        {"text": "你", "start": 0.2, "end": 0.4},
        {"text": "好", "start": 0.4, "end": 0.6},
    ]
    got = group_chars_into_sentences(units, "你好。")
    assert got[0]["text"] == "你好。"
    assert (got[0]["start"], got[0]["end"]) == (0.2, 0.6)
