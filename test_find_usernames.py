"""Offline tests for find_usernames.py. Run: python3 -m unittest -v test_find_usernames"""

import random
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import find_usernames as f

PROFILE_HTML = (
    '<html><head><meta property="og:url" '
    'content="https://www.instagram.com/{u}/"></head><body>profile</body></html>'
)
SOFT_404_HTML = (
    "<html><body><h2>Sorry, this page isn't available.</h2>"
    "<p>The link you followed may be broken.</p></body></html>"
)
LOGIN_HTML = "<html><body>Log in to Instagram</body></html>"


class ClassifyResponseTests(unittest.TestCase):
    def test_404_is_not_found(self):
        label, _ = f.classify_response("abc", 404, None, "")
        self.assertEqual(label, "not_found")

    def test_200_soft_404_is_not_found(self):
        label, _ = f.classify_response("abc", 200, None, SOFT_404_HTML)
        self.assertEqual(label, "not_found")

    def test_200_soft_404_curly_apostrophe(self):
        body = "Sorry, this page isn\u2019t available."
        label, _ = f.classify_response("abc", 200, None, body)
        self.assertEqual(label, "not_found")

    def test_200_real_profile_is_taken(self):
        label, _ = f.classify_response("abc", 200, None, PROFILE_HTML.format(u="abc"))
        self.assertEqual(label, "taken")

    def test_profile_match_is_case_insensitive(self):
        label, _ = f.classify_response("ABC", 200, None, PROFILE_HTML.format(u="abc"))
        self.assertEqual(label, "taken")

    def test_profile_of_different_user_is_not_taken(self):
        # Page for another account must not be read as "this username is taken".
        label, _ = f.classify_response("abc", 200, None, PROFILE_HTML.format(u="xyz"))
        self.assertEqual(label, "unknown")

    def test_200_login_page_is_unknown_not_available(self):
        label, _ = f.classify_response("abc", 200, None, LOGIN_HTML)
        self.assertEqual(label, "unknown")

    def test_200_empty_body_is_unknown(self):
        label, _ = f.classify_response("abc", 200, None, "")
        self.assertEqual(label, "unknown")

    def test_redirect_is_unknown(self):
        label, reason = f.classify_response(
            "abc", 302, "https://www.instagram.com/accounts/login/", "")
        self.assertEqual(label, "unknown")
        self.assertIn("login", reason)

    def test_429_is_rate_limited(self):
        self.assertEqual(f.classify_response("abc", 429, None, "")[0], "rate_limited")

    def test_403_is_blocked(self):
        self.assertEqual(f.classify_response("abc", 403, None, "")[0], "blocked")

    def test_500_is_error(self):
        self.assertEqual(f.classify_response("abc", 500, None, "")[0], "error")


class _Handler(BaseHTTPRequestHandler):
    # path -> (status, location, body)
    routes = {}

    def do_GET(self):
        status, location, body = self.routes.get(self.path, (404, None, ""))
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):  # keep test output quiet
        pass


class CheckProfileNetworkTests(unittest.TestCase):
    """Exercise the real urllib path against a local server."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls._orig_url = f.PROFILE_URL
        f.PROFILE_URL = f"http://127.0.0.1:{cls.port}/{{username}}"

    @classmethod
    def tearDownClass(cls):
        f.PROFILE_URL = cls._orig_url
        cls.server.shutdown()

    def setUp(self):
        _Handler.routes = {
            "/free1": (404, None, ""),
            "/soft": (200, None, SOFT_404_HTML),
            "/taken1": (200, None, PROFILE_HTML.format(u="taken1")),
            "/login": (302, "https://www.instagram.com/accounts/login/", ""),
            "/limit": (429, None, ""),
            "/blocked": (403, None, ""),
        }

    def test_404_over_network(self):
        self.assertEqual(f.check_profile("free1")[0], "not_found")

    def test_soft_404_over_network(self):
        self.assertEqual(f.check_profile("soft")[0], "not_found")

    def test_profile_over_network(self):
        self.assertEqual(f.check_profile("taken1")[0], "taken")

    def test_redirect_to_login_is_not_a_hit(self):
        # urllib must NOT follow the redirect, otherwise the login page (200)
        # would be silently read as something else.
        label, reason = f.check_profile("login")
        self.assertEqual(label, "unknown")
        self.assertIn("accounts/login", reason)

    def test_429_over_network(self):
        self.assertEqual(f.check_profile("limit")[0], "rate_limited")

    def test_403_over_network(self):
        self.assertEqual(f.check_profile("blocked")[0], "blocked")

    def test_network_failure_is_error(self):
        orig = f.PROFILE_URL
        try:
            f.PROFILE_URL = "http://127.0.0.1:1/{username}"  # nothing listens here
            self.assertEqual(f.check_profile("x", timeout=2)[0], "error")
        finally:
            f.PROFILE_URL = orig


class RunLoopTests(unittest.TestCase):
    def test_only_not_found_is_a_hit(self):
        answers = {"aa": ("taken", "t"), "bb": ("unknown", "u"),
                   "cc": ("not_found", "n"), "dd": ("blocked", "b"),
                   "ee": ("not_found", "n")}

        def checker(u):
            return answers.get(u, ("taken", "t"))

        hits = f.run(count=5, min_len=2, max_len=2, style="wordlist",
                     word_list=["aa", "bb", "cc", "dd", "ee"], delay=0,
                     max_checks=5, output=None, checker=checker)
        self.assertEqual(hits, ["cc", "ee"])

    def test_unknown_results_never_become_hits(self):
        hits = f.run(count=3, min_len=2, max_len=2, style="wordlist",
                     word_list=["aa", "bb", "cc"], delay=0, max_checks=10,
                     output=None, checker=lambda u: ("unknown", "?"),
                     max_consecutive_unknown=100)
        self.assertEqual(hits, [])

    def test_stops_when_instagram_keeps_refusing(self):
        calls = []

        def checker(u):
            calls.append(u)
            return ("blocked", "403")

        words = [f"w{i:03d}" for i in range(100)]
        f.run(count=3, min_len=2, max_len=5, style="wordlist", word_list=words,
              delay=0, max_checks=100, output=None, checker=checker,
              max_consecutive_unknown=5)
        self.assertEqual(len(calls), 5)


class UsernameRuleTests(unittest.TestCase):
    def test_rules(self):
        self.assertTrue(f.is_valid_username("a.b_c"))
        self.assertFalse(f.is_valid_username(".ab"))
        self.assertFalse(f.is_valid_username("a..b"))
        self.assertFalse(f.is_valid_username("ab."))
        self.assertFalse(f.is_valid_username("a-b"))


if __name__ == "__main__":
    unittest.main()
