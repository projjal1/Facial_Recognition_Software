# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Django 5.2 LTS web portal (project package `chatapp`) that wraps OpenCV face recognition into a log-based entry/exit surveillance system. Originally a hackathon proof-of-work on Django 2.2, upgraded 2026-08-01. Runs on Python 3.12; pins are in `requirements.txt`.

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

```bash
python manage.py evaluate_recognition
```

Holds out part of each person's images, trains in memory (never touching `trainer.yml`), and prints correct/wrong/rejected rates across a range of confidence thresholds. Needs at least two enrolled people.

```bash
python manage.py test
```

93 tests, roughly 90 seconds — most of that is TensorFlow loading the two bundled models at startup. Narrow the run with `python manage.py test <app>` or `python manage.py test <app>.tests.ClassName.test_method`.

Tests for the root-level modules (`security`, `admin_state`, `face_store`, `streaming`) live in the `tests/` package; each app's own tests are in its `tests.py`. Nothing needs a camera or a trained model: `camera.local_frames` is patched with a finite generator, which is the practical payoff of expressing frame sources as generators. Anything writing to disk redirects `BASE_DIR` or `FACE_IMAGE_ROOT` at a temporary directory — a test that writes to the real `admin_files/` or `face-files/` is a bug.

Application logging drops to `CRITICAL` during test runs (see the tail of `settings.py`), because several tests deliberately drive failure paths and their tracebacks are expected output. Set `DJANGO_LOG_LEVEL` to see them.

### Environment variables

Both are optional and read via `os.environ.get` — nothing is committed.

| Variable | Effect if unset |
|---|---|
| `DJANGO_SECRET_KEY` | A throwaway key is generated per process, so every restart invalidates sessions and logs all users out. Set it for anything but a quick local run. |
| `FAST2SMS_API_KEY` | `alerts.alert()` prints a notice and returns without sending; recognition otherwise continues normally. |

### Committing

Commit after each major change — a working feature, a fix, a finished refactor, a dependency or config change, a substantive docs rewrite — instead of letting work pile up. Use the **`commit-checkpoint`** skill, which holds the message format and the review steps, and run its guard before every commit:

```bash
python .claude/skills/commit-checkpoint/scripts/check_staged.py
```

The guard blocks face images, `db.sqlite3`, `trainer.yml`, hardcoded secrets and phone numbers from being staged — including via `git add -f`, which defeats `.gitignore`. Checkpoints are local; never push without being asked.

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

### Video is streamed to the browser; two paths still are not

Recognition and enrolment stream MJPEG. The chain is: `camera.local_frames()` /
`camera.remote_frames(url)` yield raw frames → `recognition.frames()` /
`enrolment.capture()` yield annotated ones → `streaming.mjpeg()` wraps each as a
multipart JPEG part → a `StreamingHttpResponse` feeds an `<img>` tag. No
JavaScript is involved; the browser consumes `multipart/x-mixed-replace` natively.

Two consequences worth knowing before changing any of it:

- **Closing the page is what stops a capture.** The write fails, Django stops
  iterating, the generator is closed, and the `finally` in `camera.py` releases
  the device. There is no `q` keypress any more, because there is no window.
- **Errors must be raised before the response starts.** Once bytes are flowing
  there is no way to send a status code. `streaming.primed()` pulls the first
  frame inside the view precisely so a missing camera or untrained model still
  renders as a normal error page. Keep new failure modes on that side of the
  line, and note `feeds.views._not_ready()` does the same job for the page that
  hosts the video.

All four capture paths work this way now — recognition, enrolment, emotion and
mask. `emotion.resources.cam.frames()` and `mask.resources.webcam.frames()`
follow the same shape as `recognition.frames()`: take a frame source, yield
annotated frames. There is no `cv2.imshow` anywhere in the project.

Because `chatapp/urls.py` includes `emotion.urls` and `mask.urls`, whose views import their `resources` modules at module scope, **TensorFlow loads the `.h5` models at server startup** — the server will not boot without `tensorflow` and `imutils` installed.

### The `sN` username convention (load-bearing)

The whole recognition pipeline hinges on an implicit naming contract:

1. `accounts.views.signup` runs `os.system("mkdir " + username)`, creating a top-level directory named after the user.
2. `face_store.enrolled_folders()` scans `face-files/` and yields `(label, path)`, where the label is `int(dirname[1:])` — so folders must be `s1`, `s2`, … Signup rejects anything else, so an account that could never be trained can no longer be created.
3. `feeds.views.start` builds `subjects = ['UNKNOWN'] + [usernames starting with 's', in DB order]` and recognition does `names[id]`.

Consequence: the displayed name is correct only while `s1..sN` exist contiguously and in `auth_user` id order. Deleting a user, or a non-`sN` username beginning with `s`, silently shifts every label. Any new top-level directory starting with `s` (other than `st*`/`sm*`) will be picked up as face data.

**Current state: the repo ships with no enrolled data at all.** No users in the DB, no `face-files/`, no `trainer.yml`. Recognition cannot run until someone signs up as `s1`, enrols images, and trains; `recognition.py` raises a clear `ValueError` until a model exists. Images now live in `face-files/<username>/` rather than in folders at the project root, and `face-files/` is gitignored in full, so re-enrolled faces cannot be committed back.

### Flat-file state (`admin_files/`)

| File | Written by | Read by |
|---|---|---|
| `logs.txt` | `identify.py` / `webcam.py` append `<name> logged at <timestamp>` | `accounts.views.logs` (POST clears it) |
| `mobile_no.txt` | `accounts.views.profile` | `identify.py` / `webcam.py` → passed to `alerts.alert()`. Intentionally left empty in the repo — it held a real phone number; set it through the alert-settings page, don't commit a value back. |
| `link.txt` | `feeds.views.init_url` (remote) / `init_server` (writes empty = use local webcam) | `feeds.views.start` to pick local vs remote source |
| `trained.txt` | `recog.py` writes the image count at end of training | `feeds.views.get_files_untrained` diffs it against the current count to report "pending data" |

All four files are currently empty, which every reader handles: empty `trained.txt` parses as `0`, empty `link.txt` selects the local-webcam path.

`trainer.yml` (~50 MB LBPH model) is regenerated wholesale by `recog.begin()` and is gitignored — it is derived from face images, so keep it out of commits.

### The CV pipeline

`faces.py` owns detection and crop normalisation and **both halves of the pipeline go through it** — enrolment, uploads, and recognition. That is deliberate: LBPH compares histograms of local texture, so enrolling at one scale and matching at another degrades accuracy in a way that reads as a bad threshold rather than a preprocessing mismatch. If you change the transform, everyone must re-enrol and the model must be retrained.

Detection is the SSD from `mask/resources/` (vendored for the mask app, reused here), not the Haar cascade the project started with. Stored images are already normalised crops, so `recog.py` does no detection at training time.

`tracking.py` associates detections across frames by centroid distance. Each frame casts one vote per tracked face, and `recognition.py` acts on the most-voted identity — which is why two people in view no longer share one counter.

All tuning lives in `settings.py` (`FACE_*`). The confidence thresholds are placeholders: run `manage.py evaluate_recognition` against real enrolled data and set them from its table.

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
