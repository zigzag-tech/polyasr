from cuda.evaluate_backends import evaluate


def test_same_evaluation_contract_scores_both_pinned_models():
    result = evaluate({
        "evaluation": "benchday-dictation", "evaluation_revision": "corpus-v1",
        "cohort": "en-live", "models": {"qwen": "q1", "r2t2": "r1"},
        "rows": [
            {"id": "1", "language": "en", "reference": "hello world",
             "outputs": {"qwen": "hello word", "r2t2": "hello world"}},
            {"id": "2", "language": "en", "reference": "queue the task",
             "outputs": {"qwen": "cue the task", "r2t2": "queue the task"}},
        ],
    }, measured_at=1000)
    assert result["r2t2"]["value"] == 1
    assert result["qwen"]["value"] < result["r2t2"]["value"]
    assert result["qwen"]["evaluation_revision"] == result["r2t2"]["evaluation_revision"]
    assert result["qwen"]["sample_count"] == 2
