# Debugging the spotifysongs Package Setup

Notes from a working session (Aug 31 - Sep 1, 2026) spent getting
`import spotifysongs` to work, cleaning up `pull_all.py`, fixing the
install instructions, and putting the repo on GitHub.

## The Original Problem

`import spotifysongs` followed by `spotifysongs.load()` failed in ipython.
It turned out to be three separate problems stacked on top of each other:

1. The parquet files did not exist yet (`data/` was empty).
2. `pull_all.py` had bugs that stopped it from producing them.
3. A test script sitting in the repo root imported the wrong `spotifysongs`.

## Bugs in pull_all.py

### Round 1 - stopping the script from running at all (I fixed these)

| Bug | Fix |
| --- | --- |
| Saved as `spotify_songs.csv` but `load.py` read `.parquet` | Renamed the save paths to `.parquet` |
| `compression="parquet"` is not a codec (`ArrowException`) | Changed to `compression="gzip"` |
| `sanitize_cols` was called on `df1` twice, never `df2` | Second call now passes `df2` |

### Round 2 - cleaning logic that did not match my comments (Claude fixed these)

| Bug | Fix |
| --- | --- |
| Whitespace/quote cleanup ran *after* sentinel replacement, so padded blanks like `"  None  "` survived as empty strings that `dropna()` cannot catch | Moved the text normalization into `sanitize_cols()` so it runs *before* sentinels are matched |
| The `r"^\s*$"` sentinel was dead code - `Series.replace` with a list matches exactly, not by regex | Removed it (stripping first makes it redundant) and noted the exact-match behavior in a comment |
| The CSV loop used `os.scandir()`, which yields every entry, not just the validated `*.csv` files | Loop over `csv_files` instead; also `sorted()` for deterministic order |
| `astype(bool)` is truthiness-based, so the string `"False"` would become `True` | Added `BOOL_MAP` / `to_bool()` and an optional `cast_fxn` hook in the schema |
| `genres_df` kept 130 duplicate `(track_id, genre)` pairs | Added `genres_df.drop_duplicates()` |

Cosmetic: fixed the `FUNCITONS` typo, changed `dir_exists`'s docstring from
`Returns:` to `Raises:`, and removed the now-unused `import os`.

### Round 3 - fresh clone failures (I fixed these)

| Bug | Fix |
| --- | --- |
| `data/` was gitignored including its `.gitkeep`, so a fresh clone had no `data/` and `pull_all.py` raised before writing | `SAVE_DIR.mkdir(parents=True, exist_ok=True)` (and the same for `raw_data/`) |

## Bug in load.py

The `category` dtype for `key`, `mode`, and `time_signature` did not survive
the parquet round-trip - pyarrow restores dictionary-encoded *integer*
columns as their value type. `genre` survived because its categories are
strings. Claude added a `CATEGORY_COLS` list and a re-cast loop in `load()`.

I later replaced the no-op `os.path.exists(DATA_DIR)` line with a real check
that raises a useful message telling you to run `pull_all.py` first.

## The Import Shadowing Problem

The project folder `spotifysongs/` and the package `spotifysongs/spotifysongs/`
have the same name. Python puts a script's own directory first on `sys.path`,
so a script in the repo root imports the *project folder* as an implicit
namespace package instead of the real package.

The confusing part: `python -m spotifysongs.pull_all` works from the repo
root, but `import spotifysongs; spotifysongs.load()` does not. This is
because setuptools appends its editable-install finder *after* `PathFinder`
on `sys.meta_path` - the top-level name resolves via the folder, but the
submodule falls through to the editable finder.

**Rule:** run Python from a homework directory, never the repo root. This is
now documented in `spotifysongs/README.md`.

## Import Mechanics (things I had wrong)

- `import spotifysongs` runs `__init__.py`, which *does* import `load.py`.
  But `def load():` only **defines** the function. Nothing reads a parquet
  file until you actually call `spotifysongs.load()`.
- `pull_all.py` is never imported. It is all top-level code, so importing it
  would re-run the whole ~195MB ETL on every `import`. It is a one-time build
  step you run deliberately.
- `pull_all.py` **produces** `data/*.parquet`; `load.py` **consumes** them.
  The files on disk are the handoff.

## Packaging and Install Instructions

- Removed `requirements.txt` entirely. It was a raw `pip freeze` that pinned
  `pip`/`setuptools`/`wheel`, listed `polars-runtime-32` without `polars`, and
  contained `spotifysongs-env==0.1.0`, which pip tried to fetch from PyPI and
  could never find. Dependencies now live in `pyproject.toml` only, and
  `pip install -e ./spotifysongs` pulls the whole tree correctly.
- **`pip freeze` is unsafe in a conda env.** Conda-installed packages come
  back as `@ file:///home/conda/...` build paths that exist only on the conda
  build machine. Use `pip list --format=freeze` instead (see
  `get_requirements.md`).
- Renamed the distribution from `spotifysongs-env` to `spotifysongs` so it
  matches the import name.
- Set `requires-python = ">=3.12"`. The code calls
  `Path.glob(..., case_sensitive=False)`, which was added in Python 3.12, so
  the old `>=3.11` would have allowed a broken install.
- Removed `ipython` from `dependencies` - nothing in the package imports it.
- Added `conda activate` and the "run from a homework directory" warning to
  the READMEs, and removed a leftover assignment checklist from the top of
  `spotifysongs/README.md`.
- Changed `pip install ./spotifysongs` to `pip install -e ./spotifysongs`.
  Without `-e`, pip copies only the `.py` files into site-packages, so
  `DATA_DIR` resolves to `<site-packages>/data` and both scripts break.

The documented install was verified end to end in a fresh clone with a fresh
venv: it produces 36,342 songs and 44,498 track/genre pairs.

## Rewriting Git History

`git rm --cached` removed the raw CSVs from the working tree but left the
blobs in history - `.git` was 95MB, and `kaggle_1m_songs.csv` is 167.7 MiB,
over GitHub's hard 100 MiB per-file limit. The push would have been rejected.

Claude backed the repo up to `~/Code/python-419-backup` and ran:

```
git filter-repo --path spotifysongs/raw_data --invert-paths --force
```

Result: `.git` went from 95M to 164K. All commit messages, authors, and
timestamps were preserved; per-file history still walks correctly; `git fsck`
is clean. One commit disappeared (`Moved raw data to separate directory`)
because it contained nothing but the two CSVs. All hashes after the initial
commit changed, which is fine since nothing had been pushed.

**Lesson:** `git rm --cached` does not shrink history. Only a rewrite does.

## Connecting to GitHub

You can attach an existing repo to a new remote without losing history - no
cloning or copy-pasting:

```
git remote add origin git@github.com:Olivia-B18/python-419.git
git push -u origin main
```

The GitHub repo must be created **empty** (no README, no .gitignore, no
license), otherwise the two histories share no common ancestor and the push
is refused.

Auth notes:

- GitHub stopped accepting account passwords over HTTPS in 2021. An HTTPS
  prompt wants a Personal Access Token, not your password.
- Use `git remote set-url origin <new-url>` to switch HTTPS -> SSH. There is
  no need to remove and re-add the remote.
- Tell the prompts apart: `Password for 'https://...'` is GitHub asking for a
  token. `Enter passphrase for key '/Users/...'` is your local private key,
  which never travels to GitHub.
- The SSH URL uses a colon: `git@github.com:Owner/repo.git`

## Still Outstanding

- `mkdir spotifysongs/raw_data` in `spotifysongs/README.md` fails on a rerun;
  should be `mkdir -p`.
- The README says "move the downloaded CSV's" but Kaggle ships zip archives -
  the unzip step is not mentioned.
- `load.py` checks that `data/` exists but not that the parquet files do, so
  an interrupted `pull_all.py` still produces a raw traceback.
- `~/.ssh/config` still has five `Host github.com` blocks. Four of them say
  `User Olivia-B18`; GitHub requires `User git`. It works only because the
  correct block happens to come first.
- `~/.ssh/ssh_python_419` exists but is never used - `ssh_oaktree` is offered
  first and accepted. A dedicated per-repo key needs a `Host` alias.
- `~/Code/python-419-backup` (285M) can be deleted once the push succeeds.
