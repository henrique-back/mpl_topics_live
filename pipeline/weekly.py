"""The weekly update, end to end: run every step, then commit and push, which publishes the site (GitHub Pages).

Steps, in order: fetch_scopus, embeddings, ner, assign, basin_map, refresh_citations (on the first run of a month:
day <= 7), analysis, labels, layout, newsletter, export_site. Each runs as its own process, with its output in
logs/weekly-<date>.log. If a step fails, nothing is committed or pushed and a macOS notification names the step. A
re-baseline flagged by assign.py (runs.csv) only raises a notification: the update is published as usual.

Started every Monday by launchd (scheduling/br.ufba.mpl-weekly.plist); `--dry-run` runs every step without
committing or pushing, `--from <step>` resumes at a step.
"""
import argparse
import datetime as dt
import subprocess
import sys

import pandas as pd

import config
import taxonomy

PIPELINE = config.ROOT / 'pipeline'
STEPS = ['fetch_scopus', 'embeddings', 'ner', 'assign', 'basin_map', 'refresh_citations', 'analysis', 'labels',
         'layout', 'newsletter', 'export_site']
# What the steps write (git add); everything else stays out of the commit
OUTPUTS = ['get_data', 'NER', 'bertopic', 'graphs', 'site']


def notify(message):
    """A macOS notification (does nothing elsewhere)."""
    script = f'display notification "{message.replace(chr(34), chr(39))}" with title "Microplastics weekly update"'
    subprocess.run(['osascript', '-e', script], check=False, capture_output=True)


def run(command, log):
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    return subprocess.run(command, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='run the steps, but do not commit or push')
    parser.add_argument('--from', dest='start', choices=STEPS, help='start at this step (to resume after a failure)')
    args = parser.parse_args()

    today = dt.date.today()
    steps = STEPS[STEPS.index(args.start):] if args.start else STEPS
    steps = [s for s in steps if s != 'refresh_citations' or today.day <= 7]
    config.LOGS_DIR.mkdir(exist_ok=True)
    log_path = config.LOGS_DIR / f"weekly-{today}.log"
    with open(log_path, 'a') as log:
        log.write(f"=== Weekly update {dt.datetime.now():%Y-%m-%d %H:%M}{' (dry run)' if args.dry_run else ''}\n")
        for step in steps:
            print(f"{step}...", flush=True)
            if run([sys.executable, str(PIPELINE / f'{step}.py')], log):
                notify(f"Failed at {step}; nothing was published. See {log_path.name}.")
                sys.exit(f"{step} failed; see {log_path}")
            if step == 'assign' and taxonomy.RUNS.exists():
                last = pd.read_csv(taxonomy.RUNS).iloc[-1]
                if str(last.get('rebaseline_due')).lower() == 'true':
                    notify("A re-baseline of the topics is due (see runs.csv); publishing anyway.")
        if args.dry_run:
            print(f"Dry run: nothing committed. Log: {log_path}")
            return
        if run(['git', 'add', '--', *OUTPUTS], log):
            notify("git add failed; nothing was published.")
            sys.exit("git add failed")
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=config.ROOT).returncode == 0:
            print("Nothing changed")
            return
        if run(['git', 'commit', '-m', f"Weekly update {today}"], log) or run(['git', 'push'], log):
            notify("Commit or push failed; the site was not updated.")
            sys.exit(f"commit or push failed; see {log_path}")
    print(f"Published. Log: {log_path}")


if __name__ == '__main__':
    main()
