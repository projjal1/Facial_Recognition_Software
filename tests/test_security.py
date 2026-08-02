"""Camera URL validation.

This is the check that stops an operator-supplied address being used to make
the server fetch something it should not, so the negative cases matter as much
as the positive one.
"""

from django.test import SimpleTestCase, override_settings

from security import UnsafeCameraURL, validate_camera_url


class ValidateCameraURLTests(SimpleTestCase):

    def assertRejected(self, url):
        with self.assertRaises(UnsafeCameraURL):
            validate_camera_url(url)

    def test_requires_a_url(self):
        self.assertRejected('')
        self.assertRejected('   ')

    def test_rejects_non_http_schemes(self):
        # file:// in particular would turn this into a local file reader.
        for url in ['file:///etc/passwd', 'ftp://camera/', 'gopher://camera/']:
            self.assertRejected(url)

    def test_rejects_a_url_with_no_host(self):
        self.assertRejected('http:///shot.jpg')

    def test_rejects_loopback(self):
        # The server itself, reachable under several spellings.
        self.assertRejected('http://127.0.0.1:8000/shot.jpg')
        self.assertRejected('http://localhost/shot.jpg')

    def test_rejects_link_local(self):
        # 169.254.169.254 is the cloud metadata endpoint - the classic target.
        self.assertRejected('http://169.254.169.254/latest/meta-data/')

    def test_allows_a_camera_on_the_local_network(self):
        # The normal deployment. Blocking private ranges outright would break
        # the feature this check exists to protect.
        url = 'http://192.168.1.50:8080/shot.jpg'
        self.assertEqual(validate_camera_url(url), url)
        self.assertEqual(validate_camera_url('https://10.0.0.7/video'),
                         'https://10.0.0.7/video')

    def test_returns_the_url_stripped(self):
        self.assertEqual(validate_camera_url('  http://10.0.0.7/v  '),
                         'http://10.0.0.7/v')

    @override_settings(CAMERA_URL_ALLOWED_HOSTS=['cam.example.org'])
    def test_allowlist_replaces_the_other_checks(self):
        # With an allowlist configured, membership is the whole decision - so a
        # listed host is accepted without a DNS lookup, and anything else is
        # refused even though it would otherwise pass.
        self.assertEqual(validate_camera_url('http://cam.example.org/s.jpg'),
                         'http://cam.example.org/s.jpg')
        self.assertRejected('http://192.168.1.50/s.jpg')

    def test_unsafe_camera_url_is_a_value_error(self):
        # The views catch ValueError, so this relationship is load-bearing.
        self.assertTrue(issubclass(UnsafeCameraURL, ValueError))
