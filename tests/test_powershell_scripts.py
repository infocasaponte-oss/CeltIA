from pathlib import Path


def test_autostart_does_not_assign_to_reserved_pid_variable():
    text = Path("scripts/autostart.ps1").read_text(encoding="utf-8").lower()
    assert "$pid =" not in text
    assert "$processid =" in text
