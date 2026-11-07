"""Full rebuild only: clean a complete SCOPUS export (get_data/scopus_data.csv) into get_data/alldata.csv.

This rewrites alldata.csv and changes its row order, which invalidates every row-aligned file downstream. For the
weekly updates use fetch_scopus.py, which appends only the new papers.
"""
import re

import pandas as pd

import config

# A publisher's copyright statement ("© 2022 Elsevier Ltd", "Copyright © 2023 Yadav and Kumar.", "(C) 2000 Elsevier
# Science Ltd.", ...), up to the end of the abstract or of its section: the export joins sections with "; ", so a
# statement ends at a "; " followed by a capital letter. Nearly all abstracts of the SCOPUS export have one, those of
# the Search API almost never, so it is removed to model both alike.
COPYRIGHT = re.compile(
    r'(?:This journal is\s*|Crown\s+)?'
    r'(?:©\s*)?(?:Copyright\b\s*:?\s*(?:\(C\)\s*)?©?|©|\(C\)(?=\s*\d{4})|#(?=\s*The Author))'
    r'(?:[^;]|;(?!\s*[A-Z]))*(?:;\s*|$)')
# Sub- and superscript tags of the Search API ("CO<inf>2</inf>"); the export gives the plain text ("CO2")
MARKUP = re.compile(r'</?(?:sup|inf|ani:em)>')


def normalize_abstract(text):
    """The abstract without markup or copyright statements, so that export and API abstracts read the same."""
    return COPYRIGHT.sub(' ', MARKUP.sub('', text)).strip()


def clean(data):
    # Remove duplicates based on DOI
    print("Number of duplicates based on DOI:", data.duplicated(subset=['DOI']).sum())
    data = data.drop_duplicates(subset=['DOI'])

    # Remove papers with no abstract available, or an empty one
    data = data.dropna(subset=['Abstract'])
    data['Abstract'] = data['Abstract'].map(normalize_abstract)
    data = data[~data['Abstract'].str.contains('No abstract available')]
    data = data[data['Abstract'].str.strip() != '']

    # Remove "Corrigendum" papers
    data = data[~data['Title'].str.contains('Corrigendum', na=False)]
    return data.reset_index(drop=True)


def main():
    data = pd.read_csv(config.SCOPUS_EXPORT)
    print("Length of dataset retrieved from SCOPUS:", len(data))
    data = clean(data)
    data.to_csv(config.ALLDATA, index=False)
    print("Final data set length is:", len(data))


if __name__ == '__main__':
    main()
