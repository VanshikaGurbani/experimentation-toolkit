from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EM, EN = chr(0x2014), chr(0x2013)
SUFFIXES = {".py", ".md", ".sql", ".yml", ".yaml", ".toml", ".txt"}
SKIP = {".venv", "data", ".git", ".ruff_cache", ".pytest_cache", "expkit.egg-info"}


def repo_text_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in SUFFIXES and not (set(p.relative_to(ROOT).parts) & SKIP):
            yield p


def test_no_em_or_en_dashes_anywhere():
    bad = []
    for p in repo_text_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        if EM in text or EN in text:
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"em or en dash found in: {bad}"


def test_no_secrets_or_credentials_committed():
    names = {p.name.lower() for p in repo_text_files()}
    assert ".env" not in names
    assert not any("token" in n or "kaggle.json" in n for n in names)
