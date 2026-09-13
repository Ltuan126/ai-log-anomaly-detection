from src.templates import TemplateMatcher


def test_matches_known_template_and_rejects_unknown(tmp_path):
    template_file = tmp_path / "templates.csv"
    template_file.write_text(
        "EventId,EventTemplate\nE5,[*]Receiving block[*]src:[*]dest:[*]\n",
        encoding="utf-8",
    )
    matcher = TemplateMatcher(template_file)
    assert matcher.match("Receiving block blk_12 src: node-a dest: node-b") == "E5"
    assert matcher.match("unrelated application message") is None
