"""Associating detections across frames.

The behaviour that matters is that two people in view keep separate vote
counts. A single shared counter is what produced a log entry naming whoever
happened to be in the final frame.
"""

from django.test import SimpleTestCase, override_settings

from tracking import Tracker


@override_settings(FACE_TRACK_MAX_DISTANCE=50, FACE_TRACK_MAX_MISSES=2)
class TrackerTests(SimpleTestCase):

    def test_a_face_that_barely_moves_stays_the_same_track(self):
        tracker = Tracker()
        first = tracker.update([(100, 100, 40, 40)])[0][0]
        second = tracker.update([(104, 102, 40, 40)])[0][0]
        self.assertIs(first, second)

    def test_a_face_that_jumps_too_far_becomes_a_new_track(self):
        tracker = Tracker()
        first = tracker.update([(100, 100, 40, 40)])[0][0]
        second = tracker.update([(400, 400, 40, 40)])[0][0]
        self.assertIsNot(first, second)

    def test_two_faces_get_two_tracks_and_keep_them(self):
        tracker = Tracker()
        left, right = [t for t, _ in tracker.update([(50, 50, 40, 40),
                                                     (300, 50, 40, 40)])]
        self.assertIsNot(left, right)

        again = dict(tracker.update([(52, 50, 40, 40), (302, 52, 40, 40)]))
        self.assertIn(left, again)
        self.assertIn(right, again)

    def test_votes_stay_with_the_person_they_were_cast_for(self):
        # The whole point: one counter per face, not one for the frame.
        tracker = Tracker()
        for _ in range(3):
            for track, box in tracker.update([(50, 50, 40, 40), (300, 50, 40, 40)]):
                track.record('s1' if box[0] < 100 else 's2')

        leaders = sorted(t.leader() for t in tracker.tracks)
        self.assertEqual(leaders, [('s1', 3), ('s2', 3)])

    def test_a_track_survives_a_dropped_frame(self):
        # One missed detection should not discard the evidence gathered so far.
        tracker = Tracker()
        track = tracker.update([(100, 100, 40, 40)])[0][0]
        track.record('s1')

        tracker.update([])
        again = tracker.update([(100, 100, 40, 40)])[0][0]

        self.assertIs(again, track)
        self.assertEqual(track.leader(), ('s1', 1))

    def test_a_track_is_dropped_once_it_has_been_gone_too_long(self):
        tracker = Tracker()
        tracker.update([(100, 100, 40, 40)])
        for _ in range(3):
            tracker.update([])
        self.assertEqual(tracker.tracks, [])

    def test_leader_reports_the_most_voted_identity(self):
        tracker = Tracker()
        track = tracker.update([(10, 10, 40, 40)])[0][0]
        for name in ['s1', 's1', None, 's1', None]:
            track.record(name)
        self.assertEqual(track.leader(), ('s1', 3))

    def test_unrecognised_frames_vote_for_nobody(self):
        tracker = Tracker()
        track = tracker.update([(10, 10, 40, 40)])[0][0]
        for _ in range(4):
            track.record(None)
        self.assertEqual(track.leader(), (None, 4))

    def test_reset_clears_the_votes_without_losing_the_track(self):
        tracker = Tracker()
        track = tracker.update([(10, 10, 40, 40)])[0][0]
        track.record('s1')
        track.reset()
        self.assertEqual(track.leader(), (None, 0))
        self.assertIn(track, tracker.tracks)
