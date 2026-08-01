---
name: commit-checkpoint
description: Commit finished work in this repository as a checkpoint, after running a guard that blocks face images, the database, the trained model, and hardcoded secrets from being staged. Use this whenever a coherent unit of work is complete - a feature, a bug fix, a refactor, a dependency or config change, a docs rewrite - and whenever the user says commit, checkpoint, save progress, or "that works". Run it proactively at the end of each major change rather than waiting to be asked; uncommitted work piling up in this repo is what lets sensitive files drift back in.
---

# Commit checkpoint

## Why this repo needs a gate

This project's git history had to be rewritten once already, to purge 844 enrolled
face images, a Django `SECRET_KEY`, a live SMS API key, a phone number, and a
database full of password hashes. A rewrite is expensive and it never fully works:
anyone who cloned beforehand still has the data.

The cheapest defence is small, frequent commits that are each reviewed before they
land. That is what this skill is for. The guard script does the mechanical checking
so the review stays quick.

## What counts as a major change

Commit when the tree is in a state you would be happy to return to:

- a feature works end to end
- a bug is fixed
- a refactor is complete and nothing is half-renamed
- dependencies, settings, or `.gitignore` changed
- documentation was substantively rewritten
- you are about to do something risky and want a restore point

Hold off while the work is still in motion: half-finished edits, debug prints,
scaffolding you intend to delete, experiments you expect to revert.

A useful test: can you describe the change in one sentence without using "and"?
If yes, it is one unit and it is ready. If the sentence needs three clauses, you
are probably looking at three commits, and splitting them will make the history
far easier to read later.

## Procedure

1. **Look at what changed.** `git status --short` and `git diff`. Read it; do not
   skim. This is the step that catches an accidental `admin_files/mobile_no.txt`
   or a stray capture folder.

2. **Stage deliberately.** Prefer naming paths - `git add README.md requirements.txt` -
   over `git add -A`. Blind staging is how the face images got committed in the
   first place. If you do stage everything, review `git diff --cached` afterwards.

3. **Run the guard:**
   ```bash
   python .claude/skills/commit-checkpoint/scripts/check_staged.py
   ```
   It exits non-zero and explains itself if anything sensitive is staged. Unstage
   offenders with `git restore --staged <path>`, and if the file should never be
   tracked, add it to `.gitignore` in the same commit.

4. **Check the Python still parses**, since dependencies are often not installed
   and a broken checkpoint is worse than none:
   ```bash
   python -m compileall -q <changed .py files>
   ```

5. **Commit** using the message format below.

6. **Stop there. Do not push.** See "Pushing" below.

## Message format

Subject line: imperative mood, under 72 characters, no trailing period. Body:
explain *why* the change was made, since the diff already shows what changed.
Wrap at 72 columns. End with the co-author trailer.

**Example 1** - a focused fix:
```
Skip SMS alert when no API key is configured

alerts.alert() ran inside the recognition loop with no guard, so a missing
key raised and killed the capture. Return early with a notice instead;
recognition continues without alerting.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

**Example 2** - a change that spans files but is still one idea:
```
Pin dependencies and document the supported Python range

Django 2.2 and numpy 1.21 cap this project at Python 3.9, which was not
written down anywhere and made setup guesswork.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## Never commit

| Path | Why |
|---|---|
| `face-files/` (and legacy `s1/`, `s2/`, …) | Enrolled face images - biometric data of identifiable people |
| `db.sqlite3` | Real accounts and password hashes |
| `trainer.yml` | LBPH model derived from the face images |
| A literal `SECRET_KEY` in `chatapp/settings.py` | Read `DJANGO_SECRET_KEY` from the environment instead |
| A literal key in `alerts.py` | Read `FAST2SMS_API_KEY` from the environment instead |
| A number in `admin_files/mobile_no.txt` | Personal data; it is set through the UI at runtime |

All of these are gitignored, but `git add -f` and new sibling paths (`s8/`, a
second database) get past that, which is why the guard checks by pattern.

## Pushing

Checkpointing is local. Do not push as part of it.

This repository is public and its history was rewritten, so `origin` and any
existing clones have diverged in ways that make pushing a decision rather than a
formality. Ask first, every time, and never force-push without being told to.

## Branching

Commit on the current branch. This project's workflow is direct-to-`master` and
checkpoints are meant to be a running record of the session, so spinning up a
branch per checkpoint would fragment that. If the user asks for a branch, make
one - otherwise stay put.
