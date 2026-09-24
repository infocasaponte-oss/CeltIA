from celtia.decision.json_safety import validate_json_depth


def test_json_safety_accepts_shared_container_references():
    shared={"value":[1,2,3]}
    validate_json_depth({"left":shared,"right":shared})


def test_json_safety_rejects_direct_cycle():
    value=[]
    value.append(value)
    try:
        validate_json_depth(value)
        assert False
    except ValueError as exc:
        assert "cyclic JSON container" in str(exc)


def test_json_safety_rejects_indirect_cycle():
    outer={}
    inner=[outer]
    outer["inner"]=inner
    try:
        validate_json_depth(outer)
        assert False
    except ValueError as exc:
        assert "cyclic JSON container" in str(exc)


def test_json_safety_enforces_depth_boundary_without_recursion():
    root=[]
    cursor=root
    for _ in range(4):
        child=[]
        cursor.append(child)
        cursor=child
    validate_json_depth(root,max_depth=4)
    cursor.append([])
    try:
        validate_json_depth(root,max_depth=4)
        assert False
    except ValueError as exc:
        assert "exceeds 4 levels" in str(exc)


def test_json_safety_rejects_non_positive_max_depth():
    for value in (0,-1):
        try:
            validate_json_depth({},max_depth=value)
            assert False
        except ValueError as exc:
            assert "positive" in str(exc)
