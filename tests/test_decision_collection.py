import asyncio
from types import SimpleNamespace

from scripts.collect_decision_eval_results import collect_one, load_datasets


class FakeRuntime:
    def __init__(self):
        self.calls=[]

    async def decide_with_usage(self, context, questions):
        self.calls.append((context, questions))
        return [
            SimpleNamespace(
                decision="think",
                confidence=.8,
                abstained=False,
                suspected_ood=False,
                abstention_reason=None,
                normalized_entropy=.3,
                margin=.5,
            ),
            SimpleNamespace(probabilities={"false":.9,"true":.1}),
        ], {"prompt_tokens":2,"completion_tokens":2,"total_tokens":4,"models":["test-model"]}


def test_collect_one_matches_shadow_routing_shape():
    runtime=FakeRuntime()
    row={"text":"Analiza este problema paso a paso.","expected":"think","ood":False}
    item=asyncio.run(collect_one(runtime,row))

    assert item["text"] == row["text"]
    assert item["cde"] == "think"
    assert item["confidence"] == .8
    assert item["abstained"] is False
    assert item["suspected_ood"] is False
    assert item["ood_probability"] == .1
    assert item["ood_classifier_decision"] is False
    assert item["expected"] == "think"
    assert item["expected_ood"] is False
    assert item["models_used"] == ["test-model"]
    context,questions=runtime.calls[0]
    assert context["user_message"] == row["text"]
    assert "heuristic_route" not in context
    assert context["input_chars"] == len(row["text"])
    assert context["long_context_chars"] == 12000
    assert questions == [
        {
            "id":"route",
            "prompt":"Select the most appropriate CeltIA execution route.",
            "type":"choice",
            "options":["fast","think","code","agent","long"],
        },
        {
            "id":"route_ood",
            "prompt":"Is this input outside the CeltIA routing domain?",
            "type":"boolean",
        },
    ]


def test_load_datasets_combines_and_rejects_duplicate_text(tmp_path):
    first=tmp_path/"one.jsonl"
    second=tmp_path/"two.jsonl"
    first.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    second.write_text('{"text":"two","expected":null,"ood":true}\n',encoding="utf-8")
    rows=load_datasets([str(first),str(second)])
    assert [row["text"] for row in rows] == ["one","two"]

    second.write_text('{"text":"one","expected":null,"ood":true}\n',encoding="utf-8")
    try:
        load_datasets([str(first),str(second)])
        assert False
    except ValueError as exc:
        assert "duplicate benchmark text" in str(exc)
