# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Django 2.2 web portal (project package `chatapp`) that wraps OpenCV face recognition into a log-based entry/exit surveillance system. Originally a hackathon proof-of-work. There is no `requirements.txt`; the committed `__pycache__` is CPython 3.7.

Dependencies used by the code: `django`, `opencv-contrib-python` (the `cv2.face` LBPH API is in contrib, not base `opencv-python`), `numpy`, `Pillow`, `requests`, `tensorflow`, `imutils`.

## Commands

```bash
python manage.py runserver
```

```bash
python manage.py migrate
```

```bash
python manage.py createsuperuser
```

Tests: every `*/tests.py` is an unmodified stub — there is no test suite. The mechanism if one is added is `python manage.py test <app>` / `python manage.py test <app>.tests.ClassName.test_method`.

### Environment variables

Both are optional and read via `os.environ.get` — nothing is committed.

| Variable | Effect if unset |
|---|---|
| `DJANGO_SECRET_KEY` | A throwaway key is generated per process, so every restart invalidates sessions and logs all users out. Set it for anything but a quick local run. |
| `FAST2SMS_API_KEY` | `alerts.alert()` prints a notice and returns without sending; recognition otherwise continues normally. |

**Always run `manage.py` from the repository root.** Nearly every path in the CV code is relative to the process CWD (`trainer.yml`, `haarcascade_frontalface_default.xml`, `admin_files/*.txt`, `emotion/resources/*`, `mask/resources/*`, and the per-user image folders).

## Architecture

### Apps and what each owns

| App | Role |
|---|---|
| `accounts` | Login/signup/logout, superuser-only alert-settings and system-log pages, per-user "about" page. Also owns `home.html`'s entry view (`views.base`). |
| `records` | Face **enrollment** — capture training images into the user's folder (local webcam, remote MJPEG-style URL, or file upload). |
| `feeds` | Superuser-only **training + live recognition** control panel. |
| `emotion` | Experimental Keras FER (7 emotions) over the local webcam. |
| `mask` | Experimental mask/no-mask/improper detection (Caffe SSD face detector + MobileNetV2 classifier). |

All `models.py` files are empty stubs and all `migrations/` contain only `__init__.py`. **The only DB model in use is Django's built-in `User`.** All application state lives in flat files (see below).

### The CV modules at repo root

`start.py`, `remote_start.py` (enroll: capture 15 face crops), `recog.py` (train LBPH → `trainer.yml`), `identify.py`, `webcam.py` (recognize: local webcam / remote URL), `alerts.py` (SMS). These are plain modules imported directly by views — not Django apps.

### Critical: CV runs synchronously in the request cycle

`records.views.get_face`, `feeds.views.start`, `feeds.views.train`, `emotion.views.detect`, and `mask.views.start_mask` call blocking OpenCV loops that open a `cv2.imshow` window **on the machine running the dev server**. The HTTP request does not return until the loop ends (user presses `q`, or the frame/confidence counters trip). This makes the app single-user and desktop-bound; keep it in mind before "fixing" anything that looks like a hang.

Because `chatapp/urls.py` includes `emotion.urls` and `mask.urls`, whose views import their `resources` modules at module scope, **TensorFlow loads the `.h5` models at server startup** — the server will not boot without `tensorflow` and `imutils` installed.

### The `sN` username convention (load-bearing)

The whole recognition pipeline hinges on an implicit naming contract:

1. `accounts.views.signup` runs `os.system("mkdir " + username)`, creating a top-level directory named after the user.
2. `recog.py::getImagesAndLabels` scans top-level dirs, **skipping any not starting with `s`** and skipping the `st*`/`sm*` prefixes (this is how `static/` and `staticfiles/` are excluded). The LBPH label is `int(dirname[1:])` — so folders must be `s1`, `s2`, … and a user named anything else is silently never trained.
3. `feeds.views.start` builds `subjects = ['UNKNOWN'] + [usernames starting with 's', in DB order]` and recognition does `names[id]`.

Consequence: the displayed name is correct only while `s1..sN` exist contiguously and in `auth_user` id order. Deleting a user, or a non-`sN` username beginning with `s`, silently shifts every label. Any new top-level directory starting with `s` (other than `st*`/`sm*`) will be picked up as face data.

**Current state: the repo ships with no enrolled data at all.** No users in the DB, no `s*/` image folders, no `trainer.yml`. Recognition cannot run until someone signs up as `s1`, enrols images, and trains. `identify.py`/`webcam.py` will fail at `recognizer.read('trainer.yml')` until a model exists — train via the feeds panel first. `s[0-9]*/` is now gitignored, so re-enrolled faces will not be committed back.

### Flat-file state (`admin_files/`)

| File | Written by | Read by |
|---|---|---|
| `logs.txt` | `identify.py` / `webcam.py` append `<name> logged at <timestamp>` | `accounts.views.logs` (POST clears it) |
| `mobile_no.txt` | `accounts.views.profile` | `identify.py` / `webcam.py` → passed to `alerts.alert()`. Intentionally left empty in the repo — it held a real phone number; set it through the alert-settings page, don't commit a value back. |
| `link.txt` | `feeds.views.init_url` (remote) / `init_server` (writes empty = use local webcam) | `feeds.views.start` to pick local vs remote source |
| `trained.txt` | `recog.py` writes the image count at end of training | `feeds.views.get_files_untrained` diffs it against the current count to report "pending data" |

All four files are currently empty, which every reader handles: empty `trained.txt` parses as `0`, empty `link.txt` selects the local-webcam path.

`trainer.yml` (~50 MB LBPH model) is regenerated wholesale by `recog.begin()` and is gitignored — it is derived from face images, so keep it out of commits.

### Recognition tuning constants

Duplicated between `identify.py` (local) and `webcam.py` (remote) and deliberately different: confidence threshold `< 53` local vs `< 48` remote; 60 consecutive valid frames → write a log line and `sleep(3)`; 150 invalid frames → `alerts.alert()` SMS. Changing behavior usually means editing both files.

### Templates

`chatapp/templates/base.html` holds the nav and Bootstrap CDN links; `home.html` extends it; every app template extends `home.html` (not `base.html`). App templates live in `<app>/templates/` (flat, no per-app namespace directory) and are resolved by `APP_DIRS`. The nav is the de-facto authorization layer: `user.is_superuser` gates the feeds/logs/alert-settings links, regular users get face registration. **Views themselves have no `@login_required` or superuser checks** — several will raise on anonymous access (e.g. `listdir('')`).

## Known hazards in the committed code

These are pre-existing, not things to fix incidentally, but do not copy the patterns:

- `chatapp/settings.py` still has `DEBUG = True` and `ALLOWED_HOSTS = ['*']`.
- Secrets are no longer in source (see Environment variables above) — **but the old `SECRET_KEY`, fast2sms API key, and phone number remain in git history.** The API key has been rotated. Never re-add a literal value to `settings.py` or `alerts.py`.
- `db.sqlite3` is committed. It has been emptied of all accounts, sessions and admin-log rows (schema, migration ledger, content types and permissions kept), so it is now equivalent to a fresh `migrate`. Earlier commits still contain the original users and password hashes.
- The face images in `s1/`–`s7/` (844 files), `trainer.yml`, and the entry timestamps in `admin_files/logs.txt` have been deleted from the working tree — **but all of it is still in git history.** This is real biometric data of identifiable people; only a history rewrite actually removes it.
- `accounts.views.signup` passes the raw username to `os.system("mkdir ...")`.
- `records.views` renders `page2.html`; `feeds.views` renders `model_detection.html`/`start_captures.html` — mismatched template names between apps are easy to get wrong.
