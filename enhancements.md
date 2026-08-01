# Enhancements backlog

Working notes, not tracked in git. Drafted 2026-08-01 from a read of the codebase;
every claim below was checked against the source, and file:line references point at
the code as it stood that day.

Nothing here has been implemented. Items are ordered by impact within each section.

**Contents**
- [Part 1 - Application](#part-1---application)
  - [Security](#security)
  - [Bugs](#bugs)
  - [Architecture](#architecture)
  - [Modernization](#modernization)
  - [Quality](#quality)
- [Part 2 - Face recognition pipeline](#part-2---face-recognition-pipeline)
- [Suggested order](#suggested-order)

---

## Part 1 - Application

### Security

- [ ] **Add authentication to every view.** There are zero occurrences of
  `login_required`, `user_passes_test` or `is_superuser` in any `.py` file - the only
  role check is in `chatapp/templates/base.html:87`, which merely hides nav links. An
  anonymous visitor can read the entry log at `/accounts/logs/`, change the alert phone
  number at `/accounts/profile/`, or hit `/feed_detect/handler-start/` to start the
  camera and recognition loop on the host machine. Apply `@login_required` broadly and
  `@user_passes_test(lambda u: u.is_superuser)` to the `feeds` views and the admin pages.

- [ ] **Fix command injection in signup.** `accounts/views.py:45` runs
  `os.system("mkdir " + field1)` on the raw POST username. Signup calls
  `User.objects.create_user` directly rather than through a form, so Django's username
  validator never runs and the string reaches the shell unfiltered. Replace with
  `os.makedirs(...)` and validate the username against the `sN` pattern the pipeline
  requires anyway.

- [ ] **Constrain server-side URL fetching.** `remote_start.record_vid` and
  `webcam.remote` fetch an arbitrary user-supplied URL (`records/views.py:44`,
  `feeds/views.py:17`). With authentication missing this is a usable SSRF against
  internal addresses. Allowlist hosts, or restrict to a camera URL set in configuration.

- [ ] **Turn off debug for anything non-local.** `DEBUG = True` and
  `ALLOWED_HOSTS = ['*']` in `chatapp/settings.py` expose full tracebacks, including
  local variables, on any error.

### Bugs

- [ ] **`webcam.py:106` calls `cv2.release()`, which does not exist.** Remote
  recognition raises `AttributeError` on every exit. There is no capture object to
  release in that function either - it reads frames over HTTP with `requests` - so the
  line should simply be deleted.

- [ ] **Variable collision corrupts the face box.** Inside `for (x,y,w,h) in faces:`,
  `identify.py:67` reassigns `x` to a datetime, clobbering the face's x-coordinate for
  the remainder of that iteration. Same defect at `webcam.py:67`. Rename to `now`.

- [ ] **`logout` returns `None` on GET.** `accounts/views.py:24` has no `else` branch,
  so a GET raises "view didn't return an HttpResponse".

- [ ] **Replace three bare `except:` blocks** (`feeds/views.py:52`, `feeds/views.py:97`,
  `records/views.py:50`). They swallow every exception including `KeyboardInterrupt`.
  The training one reports "Some error occurred" with no traceback, which makes failures
  undiagnosable.

### Architecture

- [ ] **Stream video to the browser instead of blocking the request.** The
  `cv2.imshow` loop runs inside the request cycle and opens a window on the *server*,
  which is what makes the app single-user, desktop-bound and unusable when deployed. A
  `StreamingHttpResponse` yielding MJPEG frames fixes all three. Highest-value change in
  this document.

- [ ] **Replace the `sN` convention with a real model.** A `Person` model holding a FK
  to `User` and an explicit integer label removes the "usernames must be contiguous and
  in database order" coupling that currently makes `names[id]` an `IndexError` waiting
  to happen.

- [ ] **Move the entry log into the database.** `admin_files/logs.txt` is appended from
  the recognition loop with no locking and read wholesale into memory. A `LogEntry`
  model brings filtering, pagination, per-person history and safe concurrent writes.

- [ ] **Move tuning constants into settings.** Confidence thresholds are duplicated and
  differ between `identify.py:55` (53) and `webcam.py:55` (48), so changing behaviour
  means remembering to edit both files.

### Modernization

- [ ] **Upgrade off Django 2.2**, whose extended support ended in April 2022 - no
  security patches since. Django 4.2 LTS or 5.x is the main reason to move past
  Python 3.9.

- [ ] **Delete one dead import to unblock TensorFlow.**
  `emotion/resources/model.py:2` imports `set_session` and never uses it, and that
  single line is what pins the project below TensorFlow 2.16. Cheapest modernization
  step available.

### Quality

- [ ] **Write tests.** Every `*/tests.py` is an unmodified stub. The bugs listed above
  would have been caught by the smallest smoke test.
- [ ] **Add logging.** The codebase communicates exclusively through `print`.
- [ ] **Delete `chatapp/output.py`**, which is dead scratch code.

---

## Part 2 - Face recognition pipeline

- [ ] **Make the decision counter identity-aware.** `valid` is a single global counter
  (`identify.py:56`) incremented on any confident match, while `text = names[id]` is
  overwritten each frame. Sixty confident frames of a mixed crowd log whoever appeared
  in the *last* frame. Associate detections into tracks (centroid distance or a KCF
  tracker), accumulate votes per track, and require a majority within a sliding window.
  Consecutive-run counting also means one dropped frame discards all accumulated
  evidence.

- [ ] **Make `UNKNOWN` reachable.** `subjects[0] = 'UNKNOWN'` (`feeds/views.py:36`), but
  LBPH's `predict()` always returns the nearest *trained* label and label 0 is never
  trained - so `UNKNOWN` cannot be returned. Rejection currently rests entirely on the
  distance threshold. Construct with `cv2.face.LBPHFaceRecognizer_create(threshold=T)`,
  which returns `-1` on rejection. Note the trap: `names[-1]` silently returns the last
  person, so `-1` must be handled explicitly.

- [ ] **Calibrate the thresholds.** 53 in `identify.py:55` versus 48 in `webcam.py:55`,
  with no recorded reason for the difference. LBPH confidence is an unbounded Chi-square
  distance whose scale shifts with crop size and grid parameters, so these values do not
  survive any preprocessing change. Hold out images, sweep the threshold, plot FAR
  against FRR and choose an operating point deliberately - an access-control system
  wants a low false-accept rate, and the current rate is unknown.

- [ ] **Swap Haar for the SSD detector already in the repo.**
  `mask/resources/res10_300x300_ssd_iter_140000.caffemodel` plus `deploy.prototxt` are
  already vendored and loaded by the mask app. That detector handles angles, glasses and
  poor lighting far better than Haar, and reusing it adds no new dependency.

- [ ] **Raise the minimum face size.** `minSize=(10,10)` (`identify.py:46`) accepts
  10-pixel faces carrying no usable identity signal and invites false positives.
  `start.py:17` sets no `minSize` at all, so enrolment can bank garbage crops. Roughly
  80px is a sane floor.

- [ ] **Add alignment, fixed-size crops and illumination normalization.** Crops go
  straight from `detectMultiScale` into LBPH (`recog.py:32`) with none of the three, and
  LBPH is sensitive to all of them. Align on eye landmarks so eyes are level; resize
  every crop to a fixed 100x100 or 128x128, since variable sizes distort the per-cell
  grid histograms LBPH depends on; and apply CLAHE (`cv2.createCLAHE`) for lighting.
  Apply the identical transform at enrolment and inference or the model degrades
  silently. This is the largest accuracy gain per line of code here, and it would make
  the README's "image colour enhancement and toning" claim true - the code currently
  only calls `cvtColor`.

- [ ] **Balance the training set.** Before deletion, s5 held 671 images while s1, s2 and
  s4 held 15 - a 45:1 ratio that biases nearest-histogram matching toward the majority
  person. Cap images per person.

- [ ] **Diversify enrolment captures.** Fifteen consecutive frames from one sitting
  (`start.py:30`) are near-duplicates in a single pose and lighting condition; they
  inflate the count without adding information. Sample across sessions and require pose
  variety.

- [ ] **Gate on image quality.** Reject blurry crops using variance-of-Laplacian before
  saving.

- [ ] **Detect duplicate identities at enrolment.** The same person can currently enrol
  as both `s3` and `s6`, quietly poisoning both classes.

- [ ] **Use incremental training.** `recog.begin()` rebuilds `trainer.yml` from every
  image on disk each time. LBPH supports `update()`, turning an O(n) rebuild into an
  O(1) append for a portal where people enrol one at a time.

- [ ] **Build an evaluation harness.** There is no train/test split and no accuracy, FAR
  or FRR measurement anywhere; `admin_files/trained.txt` holds a file count and is the
  closest thing to a metric. Without a holdout set the threshold cannot be tuned,
  preprocessing changes cannot be validated, and re-enrolment regressions go unnoticed.
  This is the prerequisite for most items above.

- [ ] **Consider embeddings over LBPH.** LBPH is a 2006-era texture-histogram method.
  ArcFace via ONNX, FaceNet, or `face_recognition`/dlib give substantially better
  accuracy, work from 1-3 enrolment images instead of 15+, need no retraining at all
  (new people are new vectors), and support open-set rejection through an interpretable
  cosine-similarity threshold that is stable across preprocessing changes. TensorFlow is
  already a dependency, so this adds no new runtime - but it does add model weights and
  wants a GPU for real-time speed.

- [ ] **Add liveness detection.** A printed photo or a phone screen defeats the system
  completely. That matters here because the stated purpose is secure access points and
  automated security counters. Blink detection is the cheapest credible option;
  texture-based anti-spoofing is stronger.

---

## Suggested order

1. **Bugs and auth first** - the Part 1 security items plus the four Part 1 bugs, and
   the two pipeline bugs (identity-aware counter, reachable `UNKNOWN`). All small and
   contained.
2. **Evaluation harness** - nothing after this is measurable without it.
3. **Detector swap and preprocessing together**, then re-enrol or retrain, then
   recalibrate the thresholds against the new pipeline. Doing these separately wastes a
   calibration pass.
4. **Streaming rewrite**, which unblocks real deployment.
5. **Embeddings and liveness** - project-sized, worth planning separately.
