"""Update the citation count of every paper of alldata.csv from the Scopus Search API, in get_data/citations.csv.

Run it monthly, and after a full rebuild (clean_data.py). Papers are looked up by EID in batches of BATCH
(`EID(...) OR ...`, STANDARD view), about 200 requests for 20,000 papers. A paper Scopus does not return (its EID merged
or withdrawn) keeps its previous count and date; one with no previous row takes the `Cited by` of alldata.csv, dated
to the last fetch (config.reference_date()). Nothing is written unless every batch was retrieved.

--offline skips the API: it only adds the papers without a row, in the same way.
"""
import argparse
import datetime as dt

import pandas as pd

import config
import fetch_scopus

BATCH = 100


def scopus_counts(eids):
    """{EID: citation count} of the papers `eids` that Scopus returns."""
    secrets = config.read_secrets()['scopus']
    counts = {}
    for start in range(0, len(eids), BATCH):
        query = ' OR '.join(f'EID({eid})' for eid in eids[start:start + BATCH])
        for entry in fetch_scopus.search(query, secrets['api_key'], secrets.get('insttoken'), view='STANDARD',
                                         count=BATCH, verbose=False):
            counts[entry['eid']] = int(entry.get('citedby-count') or 0)
        print(f"{min(start + BATCH, len(eids))} / {len(eids)} papers", end='\r')
    print()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--offline', action='store_true', help="only add the papers without a row, from alldata.csv")
    args = parser.parse_args()

    data = pd.read_csv(config.ALLDATA)
    ids = config.paper_ids(data).to_numpy()
    eids = config.eids(data['Link'])
    counts = {} if args.offline else scopus_counts(eids.tolist())

    found = eids.map(counts).to_numpy()
    table = pd.DataFrame({'cited_by': found, 'retrieved_on': pd.Timestamp(dt.date.today())}, index=ids)
    missing = pd.isna(found)
    previous = config.load_citations() if config.CITATIONS.exists() else pd.DataFrame(columns=['cited_by', 'retrieved_on'])
    kept = previous.reindex(ids[missing])
    new = kept['cited_by'].isna().to_numpy()
    kept.loc[new, 'cited_by'] = data.loc[missing, 'Cited by'].to_numpy()[new]
    kept.loc[new, 'retrieved_on'] = pd.Timestamp(config.reference_date())
    table.loc[missing] = kept.to_numpy()
    table['cited_by'] = table['cited_by'].astype(int)
    config.save_citations(table)
    if not args.offline:
        print(f"{(~missing).sum()} counts refreshed; {missing.sum()} papers not returned by Scopus "
              f"({(~new).sum()} kept their previous count)")
    print(f"{new.sum()} papers added from alldata.csv, dated {config.reference_date()}")


if __name__ == '__main__':
    main()
