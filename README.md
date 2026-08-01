# Facial Recognition Software

A log-based surveillance and attendance portal built on Django and OpenCV. Faces are enrolled through the browser, trained into a local LBPH model, and recognised from a webcam — every match is written to a timestamped entry log, and repeated unrecognised faces trigger an SMS alert.

Originally produced as a proof-of-work solution in a hackathon, and implemented as a service portal for communities that need a logging system for registering time-entry and exit based on facial recognition.

---

## Features

- User registration and profile management
- Face enrolment from the server webcam, a remote camera URL, or a file upload
- Train the recognition model on demand, from the browser
- Live recognition with automatic entry logging
- SMS alert when an unrecognised face keeps appearing
- Admin control over the entire portal, with full data transparency

### Experimental

- **Face mask detection** — Caffe SSD face detector plus a MobileNetV2 mask classifier (`Mask` / `Without Mask` / `Wear Mask Properly`)
- **Emotion prediction** — a Keras CNN over 7 expressions (angry, disgust, fear, happy, neutral, sad, surprise)

Both are reachable from the navbar and run independently of the recognition pipeline.

---

## How it works

```
enrol            train                    recognise
─────            ─────                    ─────────
webcam ──┐
remote ──┼─► s<N>/*.jpg ─► LBPH ─► trainer.yml ─► predict ─┬─► confidence OK  ─► entry log
upload ──┘   (Haar crop)                                   └─► repeated fail  ─► SMS alert
```

1. **Enrol** — `haarcascade_frontalface_default.xml` detects the face in each frame and the greyscale crop is saved into a folder named after the user.
2. **Train** — every enrolled image is fed to OpenCV's LBPH recogniser, producing `trainer.yml`.
3. **Recognise** — each frame is matched against the model. Sustained confident matches write a line to the entry log; sustained failures fire an alert.

---

## Requirements

- **Python 3.7 – 3.9.** Django 2.2 does not run on Python 3.10+, so a newer interpreter will not work without upgrading the project first.
- **`opencv-contrib-python`** — required, not plain `opencv-python`. The LBPH recogniser lives in `cv2.face`, which ships only in the contrib package.
- **TensorFlow** (below 2.16) and **imutils** — needed even if you never open the experimental pages, because the URL configuration imports them at startup.

Exact pins are in [requirements.txt](requirements.txt), with the reasoning behind each constraint.

A webcam is required. The recognition window opens on the **machine running the server**, not in the browser — see [Limitations](#limitations).

---

## Setup

```bash
git clone https://github.com/projjal1/Facial_Recognition_Software.git
```

Create a virtual environment on a supported interpreter, then install the pinned dependencies:

```bash
cd Facial_Recognition_Software && python -m venv venv
```

```bash
pip install -r requirements.txt
```

The database is not committed, so create a fresh one:

```bash
python manage.py migrate
```

Then create an admin account:

```bash
python manage.py createsuperuser
```

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/ and sign in.

> Always run `manage.py` from the repository root — the recognition code resolves `trainer.yml`, the Haar cascade and `admin_files/` relative to the working directory.

### Configuration

Both variables are optional, and nothing is stored in the repository.

| Variable | Purpose | If unset |
|---|---|---|
| `DJANGO_SECRET_KEY` | Django's signing key | A throwaway key is generated per process, so every restart logs all users out. Set it for anything beyond a quick local run. |
| `FAST2SMS_API_KEY` | Sends the unrecognised-face SMS alert via [fast2sms](https://www.fast2sms.com/) | Alerts are skipped with a printed notice; recognition continues normally. |

```bash
export DJANGO_SECRET_KEY='your-key-here'
export FAST2SMS_API_KEY='your-key-here'
```

On Windows PowerShell, use `$env:DJANGO_SECRET_KEY = 'your-key-here'`.

The destination phone number for alerts is set in the browser, under **My Alert Settings**.

---

## First run: enrolling a face

The repository ships with **no enrolled data** — no accounts, no face images, no trained model. Recognition cannot run until at least one person is enrolled and the model is trained.

> **Usernames must be `s1`, `s2`, `s3`, … in that order.**
> The training step derives each person's numeric label directly from their folder name, and the recognised name is looked up by that number. A user named anything else is silently never trained, and gaps or out-of-order accounts shift every label. This is the single most important convention in the project.

1. **Sign up** as `s1`. A folder named `s1/` is created for the images.
2. **Register Face** — capture from the server webcam, point at a remote camera URL, or upload a photo. Aim for a close, well-lit shot containing mostly the face. Capture stops after 15 frames, or press `q`.
3. Repeat for `s2`, `s3`, … as needed.
4. Sign in as the **superuser** and open **Capture feeds**.
5. Choose a local or remote source, then **Train Model and Start Capture**. Training reports how many images are still pending.
6. The camera window opens. Recognised faces are logged after a sustained match; press `q` to stop.
7. Review entries under **System logs**.

---

## Project layout

| Path | Role |
|---|---|
| `chatapp/` | Django project — settings, root URLs, base templates |
| `accounts/` | Login, signup, profile, alert settings, system logs |
| `records/` | Face enrolment (webcam, remote URL, upload) |
| `feeds/` | Superuser panel: train the model and run recognition |
| `emotion/`, `mask/` | Experimental detectors with their own bundled models |
| `recog.py` | Trains the LBPH model into `trainer.yml` |
| `identify.py`, `webcam.py` | Recognition from a local webcam / remote URL |
| `start.py`, `remote_start.py` | Capture enrolment images |
| `alerts.py` | Sends the SMS alert |
| `admin_files/` | Plain-text runtime state: entry log, alert number, camera URL, trained-image count |

There are no database models — the only table in use is Django's built-in `User`, and everything else is kept in flat files.

---

## Limitations

This is a proof-of-concept from a hackathon, not a hardened deployment. Worth knowing before you rely on it:

- **The camera window opens on the server**, and the HTTP request blocks until the capture loop ends. That makes the app effectively single-user and tied to a desktop session — it will not work as-is behind a headless web server.
- **`DEBUG = True` and `ALLOWED_HOSTS = ['*']`** are set for local development. Change both before exposing the app anywhere.
- **Views have no `@login_required` decorators.** The navigation bar hides links by role, but the URLs themselves are not access-controlled.
- Recognition thresholds are tuned by hand and differ between the local and remote paths.
- LBPH training runs on the CPU; a discrete GPU only helps the experimental TensorFlow detectors.

---

## Privacy

Face images are biometric data. This repository intentionally contains none:

- Enrolled images (`s<N>/`), the trained `trainer.yml`, and `db.sqlite3` are all git-ignored.
- Alert phone numbers and API keys are read from the environment or entered in the UI — never committed.

If you fork or deploy this, keep it that way, and get consent from anyone you enrol.

---

## Concepts explored

- Haar feature-based cascade classifiers
- LBPH (Local Binary Patterns Histograms) face recogniser
- Image colour enhancement and toning

## Utility

- Automated security counters
- Secure access points
- Surveillance measures

## License

Released under the [MIT License](LICENSE).
