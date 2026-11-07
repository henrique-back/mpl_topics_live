"""The weekly newsletter: an AI-written digest of the papers added by the latest fetch_scopus.py run.

The papers of the latest date of get_data/added.csv are grouped by broad (coarse) topic. A topic with at least
config.NEWSLETTER_MIN_SECTION of them gets its own section; the others, and the papers outside every topic, share an
"Also this week" paragraph. Claude (ai.py) reads their titles and abstracts and writes an overview and the sections,
linking to a few of the papers by DOI. Links to DOIs that are not among the week's papers are removed.

Writes site/data/newsletter/<date>.json (one issue per fetch date; an existing issue is not rewritten, unless
`--redo`) and rebuilds site/data/newsletter/index.json, the list of issues, newest first.
"""
import argparse
import json
import re

import pandas as pd

import ai
import config
import taxonomy

OTHER = 'other'
TARGET_WORDS = 1000  # length of an issue (800-1200 words)

SYSTEM = """You write the weekly newsletter of a living review of the scientific literature on microplastics and \
marine plastic pollution. Each week it summarizes the papers newly indexed in Scopus, for a wide audience: the \
public, journalists and policymakers. Write clear, plain English: explain technical terms briefly or avoid them, no \
abbreviations without spelling them out, no hype, and never overstate: report what the studies found, with their \
scope (place, organism, laboratory or field), and say when results are preliminary or conflicting.

Base every statement only on the abstracts given. Synthesize: say what the week's papers show together (common \
findings, contrasts, new directions) instead of summarizing papers one by one.

Cite only a few papers: link at most 2 papers per section and at most 8 in the whole issue, choosing the most \
notable findings. Link with Markdown, using the paper's DOI exactly as given: [short description](https://doi.org/DOI).
Use Markdown only for links and occasional *emphasis*; no headings or lists inside the texts."""

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['title', 'overview', 'sections', 'also'],
          'properties': {
              'title': {'type': 'string', 'description': 'headline of the issue, at most 12 words'},
              'overview': {'type': 'string', 'description': 'the week at a glance, 100-200 words'},
              'sections': {'type': 'array', 'items': {
                  'type': 'object', 'additionalProperties': False, 'required': ['topic', 'heading', 'text'],
                  'properties': {'topic': {'type': 'integer'},
                                 'heading': {'type': 'string', 'description': 'at most 10 words'},
                                 'text': {'type': 'string', 'description': 'one paragraph'}}}},
              'also': {'type': 'string', 'description': 'one paragraph on the other papers ("" if there are none)'}}}

LINK = re.compile(r'\[([^\]]+)\]\(https?://(?:dx\.)?doi\.org/([^)\s]+)\)')


def week_papers():
    """The date of the latest fetch and its papers (paper_id), or (None, []) without a record."""
    if not config.ADDED.exists():
        return None, []
    added = pd.read_csv(config.ADDED)
    if not len(added):
        return None, []
    date = added['added_on'].max()
    return date, added.loc[added['added_on'] == date, 'paper_id'].tolist()


def keep_known_links(text, dois):
    """`text` with the links to DOIs outside `dois` replaced by their link text."""
    return LINK.sub(lambda m: m.group(0) if m.group(2).lower() in dois else m.group(1), text)


def paper_block(row):
    return (f"- DOI: {row.DOI}\n  Title: {row.Title} ({row.Year}, {row.journal})\n"
            f"  Abstract: {row.Abstract}")


def write_index():
    issues = []
    for path in sorted(config.NEWSLETTER_DIR.glob('*.json')):
        if path.name == 'index.json':
            continue
        issue = json.loads(path.read_text())
        issues.append({'date': issue['date'], 'title': issue['title'], 'n_papers': issue['n_papers']})
    issues.sort(key=lambda issue: issue['date'], reverse=True)
    (config.NEWSLETTER_DIR / 'index.json').write_text(json.dumps(issues, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--redo', action='store_true', help="rewrite the latest issue if it exists")
    args = parser.parse_args()

    date, ids = week_papers()
    path = config.NEWSLETTER_DIR / f"{date}.json"
    if not ids:
        print("No papers recorded in added.csv: no issue")
        return
    if path.exists() and not args.redo:
        print(f"The issue of {date} exists")
        return

    data = pd.read_csv(config.ALLDATA)
    data['paper_id'] = config.paper_ids(data)
    data = data[data['paper_id'].isin(ids)].rename(columns={'Source title': 'journal'}).reset_index(drop=True)
    major = taxonomy.major(taxonomy.current_version())
    fine = data['paper_id'].map(taxonomy.load_assignments(major).set_index('paper_id')['topic'])
    if fine.isna().any():
        raise SystemExit(f"{fine.isna().sum()} of the week's papers have no topic; run assign.py")
    data['topic'] = taxonomy.level_topics(fine.astype(int).to_numpy(), 'coarse', major)
    labels = taxonomy.load_labels(major).query("level == 'coarse'").set_index('topic')
    sizes = data['topic'].value_counts()
    own = [t for t in sizes.index if t != -1 and sizes[t] >= config.NEWSLETTER_MIN_SECTION]
    data['section'] = data['topic'].where(data['topic'].isin(own), OTHER)

    blocks = []
    for section, papers in data.groupby('section', sort=False):
        if section == OTHER:
            continue
        label = labels['label'].get(section, f"topic {section}")
        description = labels['description'].get(section, '')
        blocks.append(f"## Section topic={section}: {label} ({len(papers)} papers)\n{description}\n"
                      + '\n'.join(paper_block(row) for row in papers.itertuples()))
    others = data[data['section'] == OTHER]
    if len(others):
        blocks.append(f"## Also this week ({len(others)} papers from other topics)\n"
                      + '\n'.join(paper_block(row) for row in others.itertuples()))
    # Word budget: about TARGET_WORDS in all, shared by the sections after the overview and 'also' paragraphs
    per_section = max(60, (TARGET_WORDS - 150 - (100 if len(others) else 0)) // max(len(own), 1))
    prompt = (f"This week {len(data)} papers were added. Write the issue: a title, an overview, one section per "
              f"topic below (in the order given, with its topic number), and the 'also' paragraph"
              f"{'' if len(others) else ' (empty: there are no other papers)'}.\n"
              f"Length: the overview about 150 words, each section about {per_section} words"
              f"{', the also paragraph about 100 words' if len(others) else ''}. The whole issue must stay between "
              f"800 and 1200 words: select the most important findings rather than covering every paper.\n\n"
              + '\n\n'.join(blocks))
    print(f"Writing the issue of {date}: {len(data)} papers, {len(own)} sections, with {config.CLAUDE_MODEL}...")
    issue = ai.ask(prompt, SCHEMA, SYSTEM)

    dois = set(data['DOI'].dropna().str.strip().str.lower())
    sections = []
    for section in issue['sections']:
        if section['topic'] not in own:
            continue
        sections.append({'topic': section['topic'], 'label': labels['label'].get(section['topic'], ''),
                         'heading': section['heading'], 'text': keep_known_links(section['text'], dois),
                         'n_papers': int(sizes[section['topic']])})
    texts = [issue['overview'], issue['also']] + [s['text'] for s in sections]
    cited = sorted({m.group(2).lower() for text in texts for m in LINK.finditer(text)})
    titles = dict(zip(data['DOI'].str.strip().str.lower(), data['Title']))
    result = {'date': date, 'title': issue['title'].strip(), 'n_papers': len(data),
              'overview': keep_known_links(issue['overview'], dois), 'sections': sections,
              'also': keep_known_links(issue['also'], dois), 'n_also': len(others),
              'cited': [{'doi': d, 'title': titles.get(d, '')} for d in cited],
              'model': config.CLAUDE_MODEL, 'taxonomy': taxonomy.current_version()}
    config.NEWSLETTER_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    write_index()
    words = sum(len(text.split()) for text in [result['overview'], result['also']] + [s['text'] for s in sections])
    print(f"Saved {path.relative_to(config.ROOT)}: '{result['title']}', {words} words, {len(cited)} papers linked")


if __name__ == '__main__':
    main()
