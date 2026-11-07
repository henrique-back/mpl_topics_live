"""Step 1: fetch the papers added to SCOPUS since the last run and append them to get_data/alldata.csv.

Only records first loaded into Scopus after the date in get_data/scopus_last_fetch.txt (minus OVERLAP_DAYS, as a
safety margin) are requested. They are cleaned like the original export (clean_data.clean), papers already in
alldata.csv (same EID or DOI) are skipped, and the rest are appended at the end of the file. Existing rows never
move, so the later steps (embeddings.py, ner.py) only need to process the new rows. The citation count of each new
paper is also added to get_data/citations.csv, dated today (refresh_citations.py updates them later), and the
date it was added to get_data/added.csv (the papers of the weekly newsletter, newsletter.py).

Needs `api_key` (and `insttoken`, for the COMPLETE view with abstracts and all authors) in the [scopus] section of
configurations.ini. Nothing is written unless the whole search was retrieved.
"""
import argparse
import datetime as dt
import time

import pandas as pd
import requests

import config
from clean_data import clean

API_URL = 'https://api.elsevier.com/content/search/scopus'
OVERLAP_DAYS = 7
PAGE_SIZE = 25  # maximum for the COMPLETE view


def scopus_get(params, headers):
    """One page of search results, retrying on rate limits and server errors."""
    for attempt in range(5):
        response = requests.get(API_URL, params=params, headers=headers, timeout=60)
        if response.status_code != 429 and response.status_code < 500:
            break
        time.sleep(5 * 2 ** attempt)
    if not response.ok:
        raise RuntimeError(f"SCOPUS API error {response.status_code}: {response.text[:500]}")
    return response.json()['search-results']


def search(query, api_key, insttoken=None, view='COMPLETE', count=PAGE_SIZE, verbose=True):
    """All entries of a SCOPUS search, with cursor pagination (no 5000-result limit). `count` is the page size: at most
    25 for the COMPLETE view, 200 for the STANDARD view."""
    headers = {'X-ELS-APIKey': api_key, 'Accept': 'application/json'}
    if insttoken:
        headers['X-ELS-Insttoken'] = insttoken
    params = {'query': query, 'view': view, 'count': count, 'cursor': '*'}

    entries = []
    while True:
        page = scopus_get(params, headers)
        total = int(page['opensearch:totalResults'])
        # An empty result set comes back as a single entry with an 'error' key
        batch = [entry for entry in page.get('entry', []) if 'error' not in entry]
        entries += batch
        if verbose:
            print(f"Retrieved {len(entries)} / {total}", end='\r')
        if not batch or len(entries) >= total:
            break
        params['cursor'] = page['cursor']['@next']
    if verbose:
        print()
    return entries


def to_row(entry):
    """A Search API entry as a row with the columns of the SCOPUS CSV export (alldata.csv)."""
    authors = entry.get('author', [])
    affiliations = {aff.get('afid'): ', '.join(filter(None, [aff.get('affilname'), aff.get('affiliation-city'),
                                                             aff.get('affiliation-country')]))
                    for aff in entry.get('affiliation', [])}

    def author_affiliations(author):
        return [affiliations.get(afid['$'], '') for afid in author.get('afid', [])]

    keywords = entry.get('authkeywords')
    return {
        'Authors': '; '.join(a.get('authname', '') for a in authors),
        'Author full names': '; '.join(f"{a.get('surname', '')}, {a.get('given-name', '')} ({a.get('authid', '')})"
                                       for a in authors),
        'Author(s) ID': '; '.join(a.get('authid', '') for a in authors),
        'Title': entry.get('dc:title'),
        'Year': int(entry['prism:coverDate'][:4]),
        'Source title': entry.get('prism:publicationName'),
        'Cited by': int(entry.get('citedby-count') or 0),
        'DOI': entry.get('prism:doi'),
        # Same form as the export's links; the EID in it identifies the paper
        'Link': f"https://www.scopus.com/inward/record.uri?eid={entry['eid']}&partnerID=40",
        'Affiliations': '; '.join(affiliations.values()),
        'Authors with affiliations': '; '.join(', '.join([a.get('authname', '')] + author_affiliations(a))
                                               for a in authors),
        'Abstract': entry.get('dc:description'),
        'Author Keywords': '; '.join(k.strip() for k in keywords.split('|')) if keywords else None,
        'Document Type': entry.get('subtypeDescription'),
        'Source': 'Scopus',
    }


def new_papers(fetched, data):
    """The cleaned fetched papers that are not in `data` yet (matched on EID, or on DOI like clean() does).

    clean() deduplicates on DOI, which keeps a single paper without a DOI, and alldata.csv already has one, so papers
    without a DOI are dropped too: the result is the same as cleaning a full export.
    """
    fetched = clean(fetched)
    # clean() compares DOIs case-sensitively; Scopus sometimes lists one paper under DOIs that differ only in case
    fetched = fetched[~config.paper_ids(fetched).duplicated()]
    known = (config.eids(fetched['Link']).isin(config.eids(data['Link'])) | fetched['DOI'].isna()
             | config.paper_ids(fetched).isin(config.paper_ids(data)))
    return fetched[~known]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--since', help="fetch records loaded into Scopus after this date (YYYYMMDD) instead of "
                                        "the last fetch date")
    args = parser.parse_args()

    today = dt.date.today()
    if args.since:
        since = dt.datetime.strptime(args.since, '%Y%m%d').date()
    else:
        last_fetch = dt.datetime.strptime(config.SCOPUS_LAST_FETCH.read_text().strip(), '%Y%m%d').date()
        since = last_fetch - dt.timedelta(days=OVERLAP_DAYS)

    secrets = config.read_secrets()['scopus']
    query = f"{config.SCOPUS_QUERY} AND ORIG-LOAD-DATE AFT {since:%Y%m%d}"
    entries = search(query, secrets['api_key'], secrets.get('insttoken'))
    fetched = pd.DataFrame([to_row(entry) for entry in entries])
    print(f"{len(fetched)} records loaded into Scopus after {since}")

    data = pd.read_csv(config.ALLDATA)
    new = new_papers(fetched, data) if len(fetched) else fetched
    # Append without rewriting the existing rows
    if len(new):
        new[data.columns].to_csv(config.ALLDATA, mode='a', header=False, index=False)
        citations = config.load_citations() if config.CITATIONS.exists() else None
        rows = pd.DataFrame({'cited_by': new['Cited by'].to_numpy(), 'retrieved_on': pd.Timestamp(today)},
                            index=config.paper_ids(new).to_numpy())
        config.save_citations(pd.concat([citations, rows]) if citations is not None else rows)
        added = pd.DataFrame({'paper_id': config.paper_ids(new).to_numpy(), 'added_on': f"{today:%Y-%m-%d}"})
        added.to_csv(config.ADDED, mode='a', header=not config.ADDED.exists(), index=False)
    config.SCOPUS_LAST_FETCH.write_text(f"{today:%Y%m%d}\n")
    print(f"{len(new)} new papers appended; alldata.csv now has {len(data) + len(new)} papers")


if __name__ == '__main__':
    main()
