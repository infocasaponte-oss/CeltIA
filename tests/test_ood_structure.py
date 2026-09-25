import pytest

from core.ood_structure import structural_ood_reason


@pytest.mark.parametrize("text,reason", [
    ('{"pick":"a","weight":1,"signed":true}', "json_control_blob"),
    ('[{"id":"c1","win":true},{"id":"c2","win":false}]', "json_control_blob"),
    ('<verdict source="ops"><label>x</label><skip>1</skip></verdict>', "markup_control_blob"),
    ("X-Anything: value\nX-Other: yes", "header_block"),
    ("Authorization: Bearer abc", "header_block"),
    ('ts=1 level=warn msg="pinned value" actor=svc', "key_value_log"),
    ("INFO a=1 b=2 c=3", "key_value_log"),
    ("\x1b[1mBOLD\x1b[0m || flag=on", "terminal_escape_blob"),
    ("[[one]] [[two]] [[three]]", "marker_blob"),
])
def test_structural_blobs_are_flagged(text, reason):
    assert structural_ood_reason(text) == reason


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "Resume este texto en dúas frases.",
    "Why does this fail? {\"a\": 1, \"b\": [1, 2]}",
    'Corrixe este JSON e explica o erro: {"nome": "Ana", "idade": }',
    "Fix this HTML so the button is centred: <div><button>Ok</button></div>",
    "Extrae o nome desta cadea: Nome: Ana; Idade: 31.",
    "Explain what this log line means: level=error msg=\"disk full\" host=db1",
    "Nome: Ana\nCidade: Vigo\nEscribe unha presentación curta de Ana para un evento.",
    "config = {'a': 1}\nprint(config['a'])\nExplica que imprime este código.",
    "[[wiki-link]] Traduce esta expresión ao inglés.",
    "a" * 5000,
])
def test_real_requests_and_plain_text_are_not_flagged(text):
    assert structural_ood_reason(text) is None


def test_non_string_is_ignored():
    assert structural_ood_reason(None) is None
    assert structural_ood_reason(123) is None


def test_combine_marks_structural_input_as_suspected_ood_without_model_signal():
    from types import SimpleNamespace
    from core.decision_routes import combine_route_results

    route = SimpleNamespace(decision="fast", confidence=0.9, abstained=False, abstention_reason=None,
                            suspected_ood=False, normalized_entropy=0.1, margin=0.8)
    ood = SimpleNamespace(probabilities={"true": 0.01, "false": 0.99})
    assert combine_route_results(route, ood)["suspected_ood"] is False
    assert combine_route_results(route, ood, '{"pick":"a","signed":true}')["suspected_ood"] is True
    assert combine_route_results(route, ood, "Resume este texto")["suspected_ood"] is False


def _route(decision="fast", confidence=0.4, abstained=True, suspected=True, reason="low_confidence"):
    from types import SimpleNamespace
    return SimpleNamespace(decision=decision, confidence=confidence, abstained=abstained,
                           abstention_reason=reason, suspected_ood=suspected, normalized_entropy=0.9, margin=0.2)


def _ood(true_p):
    from types import SimpleNamespace
    return SimpleNamespace(probabilities={"true": true_p, "false": 1 - true_p})


def test_trusted_size_routes_long_even_when_model_hedges():
    from core.decision_routes import combine_route_results
    out = combine_route_results(_route(), _ood(0.01), "Resume os documentos adxuntos", input_chars=50000, long_context_chars=12000)
    assert out["decision"] == "long" and out["abstained"] is False
    assert out["suspected_ood"] is False and out["abstention_reason"] is None


def test_size_does_not_rescue_explicit_or_structural_ood():
    from core.decision_routes import combine_route_results
    explicit = combine_route_results(_route(), _ood(0.9), "texto", input_chars=50000, long_context_chars=12000)
    assert explicit["suspected_ood"] is True
    structural = combine_route_results(_route(), _ood(0.01), '{"pick":"a","signed":true}', input_chars=50000, long_context_chars=12000)
    assert structural["suspected_ood"] is True and structural["decision"] != "long"


def test_small_input_keeps_model_behaviour():
    from core.decision_routes import combine_route_results
    out = combine_route_results(_route(), _ood(0.01), "Resume isto", input_chars=200, long_context_chars=12000)
    assert out["decision"] is None and out["abstained"] is True and out["suspected_ood"] is True
