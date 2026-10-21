"""Shared helpers for the semantic-query experiment: concepts, multi-label prompt, parsing."""
from __future__ import annotations

import re


def load_concepts(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)["concepts"]


def concept_list_block(concepts) -> str:
    return "\n".join(f"{c['id']}. {c['desc']}" for c in concepts)


def build_prompt(concepts, modality: str) -> str:
    """modality: 'transcript' (text appended by caller) or 'audio'."""
    where = "the following meeting transcript" if modality == "transcript" else "this audio"
    return (
        f"You are given a numbered list of topics. Decide which topics are discussed in {where}.\n\n"
        f"Topics:\n{concept_list_block(concepts)}\n\n"
        # NB: no concrete numeric example — weaker audio LLMs parrot it verbatim.
        "Reply with ONLY the numbers of the topics that are discussed, separated by commas. "
        "If none apply, reply exactly: none. Do not include any other text."
    )


def parse_ids(response: str, valid_ids: set[int]) -> list[int]:
    if not response:
        return []
    if re.search(r"\bnone\b", response, re.I) and not re.search(r"\d", response):
        return []
    found = {int(x) for x in re.findall(r"\d+", response)}
    return sorted(found & valid_ids)
