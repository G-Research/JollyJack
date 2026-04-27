import os
import re

README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")


def extract_code_with_line_numbers(readme_path, start_heading):
    """Extract python code blocks, returning combined source with line numbers preserved."""
    with open(readme_path, "r") as f:
        text = f.read()

    start = text.find(start_heading)
    if start == -1:
        raise ValueError(f"Heading {start_heading!r} not found in {readme_path}")

    lines_before = text[:start].count("\n")
    section = text[start:]

    matches = list(re.finditer(r"```python\n(.*?)```", section, re.DOTALL))
    if not matches:
        raise ValueError(f"No python code blocks found after {start_heading!r}")

    current_line = 0
    parts = []
    for m in matches:
        # +1 for the ```python line itself
        block_line = lines_before + section[: m.start()].count("\n") + 1
        code = m.group(1)
        padding = block_line - current_line
        if padding > 0:
            parts.append("\n" * padding)
            current_line += padding
        parts.append(code)
        current_line += code.count("\n")

    return "".join(parts)


def test_readme_code():
    source = extract_code_with_line_numbers(README_PATH, "## How to use")
    exec(compile(source, README_PATH, "exec"), {})
