"""Daily checker for new open energy datasets.

Reads the current README.md, asks an LLM to identify open energy datasets
not yet listed there, and – if any are found – updates the README in place
and writes GitHub Actions step-output variables so the calling workflow can
open a pull request.

Required environment variables:
  OPENAI_API_KEY   – OpenAI API key (must be set).

Optional environment variables:
  GITHUB_OUTPUT    – path to the GitHub Actions output file
                     (set automatically by the runner; falls back to
                     /dev/null when running locally so the script is
                     still safe to run outside CI).
"""

import json
import os
import re
import sys

from openai import OpenAI

# ---------------------------------------------------------------------------
# README helpers
# ---------------------------------------------------------------------------

README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")


def read_readme() -> str:
    with open(README_PATH, encoding="utf-8") as fh:
        return fh.read()


def write_readme(content: str) -> None:
    with open(README_PATH, "w", encoding="utf-8") as fh:
        fh.write(content)


def extract_dataset_names(readme_content: str) -> list[str]:
    """Return every dataset name found in the README Markdown tables.

    Args:
        readme_content: Full text of README.md.

    Returns:
        List of dataset name strings (first column of each table row,
        excluding header and separator rows).
    """
    names: list[str] = []
    for line in readme_content.splitlines():
        # Skip separator rows and header rows
        if not line.startswith("|") or "---" in line:
            continue
        # First column is the dataset name
        match = re.match(r"^\|\s*([^|]+?)\s*\|", line)
        if match:
            name = match.group(1).strip()
            if name and name.lower() != "name":
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert in energy data science and open data. "
    "You have broad knowledge of publicly available energy datasets "
    "from utilities, system operators, research institutions, and "
    "government agencies world-wide."
)

_USER_TEMPLATE = """I maintain a curated list of open energy datasets in a GitHub repository.
Here are the datasets I already have:

{existing_list}

Please identify up to 5 important open energy datasets that are NOT in my current list.
Requirements:
1. Publicly accessible (immediately downloadable or free sign-up).
2. Related to energy: electricity, gas, solar, wind, grid topology, smart meters, etc.
3. Well-established, recently published, or otherwise notable.

For each dataset return a JSON object with these fields:
  "name"         – dataset name (string)
  "category"     – one of "Time Series", "Grid Topology", "Geoinformation" (string)
  "availability" – exactly "✓ immediately downloadable" or "📝 sign up needed" (string)
  "location"     – country/region with flag emoji, e.g. "🇺🇸 US" or "🌐 Global" or "—" (string)
  "link"         – canonical URL (string)
  "description"  – one sentence describing the dataset (string)

Return ONLY a valid JSON array of such objects. Do not include any other text."""


def find_new_datasets(existing_names: list[str]) -> list[dict]:
    """Call the OpenAI chat API and parse the JSON response."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    existing_list = "\n".join(f"- {n}" for n in existing_names)
    user_message = _USER_TEMPLATE.format(existing_list=existing_list)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # Strip optional Markdown code-fence wrapping
    raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)

    try:
        datasets = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Could not parse LLM response as JSON: {exc}", file=sys.stderr)
        print(f"Raw response:\n{raw}", file=sys.stderr)
        return []

    if not isinstance(datasets, list):
        print("ERROR: Expected a JSON array from LLM.", file=sys.stderr)
        return []

    return datasets


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Normalise a string for fuzzy comparison.

    Removes all non-alphanumeric characters and converts to lowercase so
    that minor punctuation or casing differences don't prevent deduplication
    (e.g. "ENTSO-E" and "ENTSOE" are treated as the same).
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def deduplicate(candidates: list[dict], existing_names: list[str]) -> list[dict]:
    """Remove candidates whose name already appears (fuzzily) in the README."""
    normalised_existing = {_normalise(n) for n in existing_names}
    unique: list[dict] = []
    for ds in candidates:
        if _normalise(ds.get("name", "")) not in normalised_existing:
            unique.append(ds)
    return unique


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {"Time Series", "Grid Topology", "Geoinformation"}


def _format_row(ds: dict) -> str:
    """Format a dataset dict as a Markdown table row.

    Args:
        ds: Dictionary with keys ``name``, ``availability``, ``location``,
            ``link``, and optionally ``description``.

    Returns:
        A Markdown table row string suitable for appending to a README table,
        e.g. ``| Name | ✓ … | 🇺🇸 US | [url](url) — description |``.
    """
    description = ds.get("description", "").strip()
    link = ds.get("link", "").strip()
    link_cell = f"[{link}]({link})" + (f" — {description}" if description else "")
    return (
        f"| {ds['name']} | {ds['availability']} | {ds['location']} | {link_cell} |"
    )


def update_readme(readme: str, new_datasets: list[dict]) -> str:
    """Append new dataset rows to the appropriate category tables in the README.

    For each dataset the target section (e.g. "## Time Series") is located.
    If the section exists, the new row is inserted just before the next ``##``
    heading.  If the section does not exist, it is created at the end of the
    file with a fresh table header.

    Args:
        readme: Current full text of README.md.
        new_datasets: List of dataset dicts (as returned by ``find_new_datasets``
            after deduplication).

    Returns:
        Updated README text with new rows added.
    """
    # Group by category
    by_category: dict[str, list[dict]] = {}
    for ds in new_datasets:
        cat = ds.get("category", "")
        if cat not in _VALID_CATEGORIES:
            cat = "Time Series"  # sensible default
        by_category.setdefault(cat, []).append(ds)

    updated = readme
    for category, datasets in by_category.items():
        new_rows = "\n".join(_format_row(ds) for ds in datasets)
        section_header = f"## {category}\n"

        if section_header not in updated:
            # Section doesn't exist – create it at the end
            updated = updated.rstrip("\n") + f"\n\n## {category}\n\n"
            updated += "| Name | Availability | Location | Link |\n"
            updated += "| ---- | ------------ | -------- | ---- |\n"
            updated += new_rows + "\n"
            continue

        section_start = updated.find(section_header)
        next_section = updated.find("\n## ", section_start + len(section_header))

        if next_section == -1:
            # Last section – append rows at the very end
            updated = updated.rstrip("\n") + "\n" + new_rows + "\n"
        else:
            # Insert rows just before the blank line that precedes the next section
            insert_pos = updated.rfind("\n", section_start, next_section) + 1
            updated = updated[:insert_pos] + new_rows + "\n" + updated[insert_pos:]

    return updated


# ---------------------------------------------------------------------------
# GitHub Actions output
# ---------------------------------------------------------------------------

def write_gha_output(key: str, value: str) -> None:
    """Write a GitHub Actions step output variable.

    Appends ``key=value`` to the file referenced by the ``GITHUB_OUTPUT``
    environment variable (set automatically by the Actions runner).  When
    running locally the variable is not set and the write is silently
    discarded via ``/dev/null``.

    Args:
        key:   Output variable name (e.g. ``"new_datasets_found"``).
        value: String value to assign.
    """
    output_file = os.environ.get("GITHUB_OUTPUT", "/dev/null")
    with open(output_file, "a", encoding="utf-8") as fh:
        fh.write(f"{key}={value}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    readme = read_readme()
    existing_names = extract_dataset_names(readme)
    print(f"Existing datasets found in README: {len(existing_names)}")

    print("Querying LLM for new dataset candidates …")
    candidates = find_new_datasets(existing_names)
    print(f"Candidates returned by LLM: {len(candidates)}")

    new_datasets = deduplicate(candidates, existing_names)
    print(f"New datasets after deduplication: {len(new_datasets)}")

    if not new_datasets:
        print("No new datasets to add.")
        write_gha_output("new_datasets_found", "false")
        return

    updated_readme = update_readme(readme, new_datasets)
    write_readme(updated_readme)
    print("README.md updated.")

    # Write a human-readable summary for the PR body
    summary_lines = [
        "This pull request was automatically created by the daily dataset checker.\n",
        "## Newly discovered datasets\n",
    ]
    for ds in new_datasets:
        desc = ds.get("description", "")
        summary_lines.append(
            f"- **{ds['name']}** ({ds.get('location', '—')}): {desc}"
        )
    summary = "\n".join(summary_lines) + "\n"

    summary_path = os.path.join(
        os.path.dirname(__file__), "..", "new_datasets_summary.md"
    )
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary)

    write_gha_output("new_datasets_found", "true")
    write_gha_output("datasets_count", str(len(new_datasets)))


if __name__ == "__main__":
    main()
