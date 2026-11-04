"""Render docs/proposal.md as site/proposal.html, styled like the project website.

    python -m figures.site_proposal

The Markdown file stays the single source; rerun this after editing it.
"""
from __future__ import annotations

import argparse
import re

import markdown

from common.config import REPO_ROOT

DOCS_URL = "https://github.com/Eternal128/ProjectX/blob/main/docs"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proposal: Seen or Solved?</title>
<meta name="description" content="Research proposal: how much of Pythia's benchmark progress over training comes from having seen the test questions.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='14' fill='%23a1ffcb' stroke='%23232323' stroke-width='6'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://api.fontshare.com">
<link rel="preconnect" href="https://cdn.fontshare.com" crossorigin>
<link href="https://api.fontshare.com/v2/css?f[]=clash-grotesk@400,500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/css/site.css">
</head>
<body>
<a class="skip" href="#doc">Skip to content</a>
<header class="doc-top">
  <a class="wordmark" href="index.html"><span class="wordmark__dot" aria-hidden="true"></span>Seen or Solved?</a>
  <a class="link" href="https://github.com/Eternal128/ProjectX/blob/main/docs/proposal.md">Source on GitHub</a>
</header>
<main class="doc" id="doc">
{body}
</main>
</body>
</html>
"""


def split_title(md_text: str) -> tuple[str, str]:
    """Split the '# Title? Subtitle' first line so the page title stays short."""
    first, _, rest = md_text.partition("\n")
    title = first.lstrip("# ").strip()
    head, q, sub = title.partition("?")
    if q and sub.strip():
        return f"<h1>{head}?</h1>\n<p class=\"doc-sub\">{sub.strip()}</p>\n", rest
    return "", md_text


def render(md_text: str) -> str:
    title_html, md_text = split_title(md_text)
    body = title_html + markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"],
                             extension_configs={"toc": {"permalink": False}})
    body = body.replace("<hr />", "")  # section breaks come from spacing on the page
    # links between docs/ files point at GitHub, since only the proposal is on the site
    body = re.sub(r'href="([\w.-]+\.md)"', rf'href="{DOCS_URL}/\1"', body)
    return PAGE.replace("{body}", body)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="docs/proposal.md")
    ap.add_argument("--out", default="site/proposal.html")
    args = ap.parse_args(argv)
    src, out = REPO_ROOT / args.src, REPO_ROOT / args.out
    out.write_text(render(src.read_text()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
