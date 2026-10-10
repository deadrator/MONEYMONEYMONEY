"""Offline tests for find_usernames.py. Run: python3 -m unittest -v test_find_usernames"""

import os
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

    def test_profile_title_marker_is_taken(self):
        body = "<html><head><title>Tango (@tango) \u2022 Instagram photos and videos</title></head></html>"
        self.assertEqual(f.classify_response("tango", 200, None, body)[0], "taken")

    def test_og_title_marker_is_taken(self):
        body = ('<meta property="og:title" content="Tango (@tango) &bull; Instagram photos" />')
        self.assertEqual(f.classify_response("tango", 200, None, body)[0], "taken")

    def test_title_of_longer_username_does_not_match(self):
        body = "<title>Tangoo (@tangoo) \u2022 Instagram photos and videos</title>"
        self.assertEqual(f.classify_response("tango", 200, None, body)[0], "unknown")

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
        self.assertEqual(sorted(hits), ["cc", "ee"])

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


class InspectTests(unittest.TestCase):
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

    def test_inspect_saves_raw_html(self):
        import os
        import tempfile
        _Handler.routes = {"/taken1": (200, None, PROFILE_HTML.format(u="taken1"))}
        with tempfile.TemporaryDirectory() as tmp:
            f.inspect_profiles(["taken1", "nobody"], out_dir=tmp)
            with open(os.path.join(tmp, "taken1.html"), encoding="utf-8") as fh:
                self.assertIn("og:url", fh.read())
            self.assertTrue(os.path.exists(os.path.join(tmp, "nobody.html")))


class UsernameRuleTests(unittest.TestCase):
    def test_rules(self):
        self.assertTrue(f.is_valid_username("a.b_c"))
        self.assertFalse(f.is_valid_username(".ab"))
        self.assertFalse(f.is_valid_username("a..b"))
        self.assertFalse(f.is_valid_username("ab."))
        self.assertFalse(f.is_valid_username("a-b"))


class ClassifyRenderedTests(unittest.TestCase):
    def test_not_found_text(self):
        label, _ = f.classify_rendered("tango", "https://www.instagram.com/tango/",
                                       "Sorry, this page isn't available.")
        self.assertEqual(label, "not_found")

    def test_profile_text(self):
        text = "Tango\n@tango\n12 posts\n340 followers\n120 following"
        label, _ = f.classify_rendered("tango", "https://www.instagram.com/tango/", text)
        self.assertEqual(label, "taken")

    def test_longer_username_is_not_a_match(self):
        text = "@tangoo\n12 posts\n340 followers"
        label, _ = f.classify_rendered("tango", "https://www.instagram.com/tango/", text)
        self.assertEqual(label, "unknown")

    def test_bare_name_and_follower_count_is_taken(self):
        # Matches the text the real page rendered for @tango.
        text = ("Log In\nSign Up\ntango\n1,294 followers\n0 following\n"
                "Wong Tango\nMeta\nAbout\nBlog\nJobs\nHelp\nAPI\nPrivacy\nTerms")
        label, _ = f.classify_rendered("tango", "https://www.instagram.com/tango/", text)
        self.assertEqual(label, "taken")

    def test_bare_name_without_stats_is_unknown(self):
        label, _ = f.classify_rendered("tango", "https://www.instagram.com/tango/", "tango\nLog In")
        self.assertEqual(label, "unknown")

    def test_login_redirect_is_unknown(self):
        label, _ = f.classify_rendered(
            "tango", "https://www.instagram.com/accounts/login/?next=/tango/",
            "@tango 12 posts 340 followers")
        self.assertEqual(label, "unknown")

    def test_blank_page_is_unknown(self):
        label, _ = f.classify_rendered("tango", "https://www.instagram.com/tango/", "")
        self.assertEqual(label, "unknown")


BROWSER_PAGES = {
    "/free1": (200, "<html><body><h2>Sorry, this page isn't available.</h2></body></html>"),
    "/taken1": (200,
                "<html><body>Loading...<script>setTimeout(function(){"
                "document.body.innerText='Taken1 @taken1  12 posts  340 followers';},300);"
                "</script></body></html>"),
    "/missing404": (404, "<html><body>not here</body></html>"),
    "/login": (200, "<html><body>Log in</body></html>"),
}


class BrowserCheckerTests(unittest.TestCase):
    """Real headless Chromium against a local server. Skipped if Chromium is missing."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _PageHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls._orig_url = f.PROFILE_URL
        f.PROFILE_URL = f"http://127.0.0.1:{cls.port}/{{username}}"
        cls.checker = f.BrowserChecker(
            headless=True,
            executable_path=os.environ.get("FIND_USERNAMES_TEST_CHROMIUM") or None,
            settle_ms=800,
        )
        cls.skip_reason = ""
        try:
            cls.checker.__enter__()
        except RuntimeError as exc:
            cls.skip_reason = str(exc)
            cls.checker = None

    @classmethod
    def tearDownClass(cls):
        f.PROFILE_URL = cls._orig_url
        if cls.checker is not None:
            cls.checker.close()
        cls.server.shutdown()

    def setUp(self):
        if self.checker is None:
            self.skipTest(f"Chromium not available: {self.skip_reason[:120]}")
        _PageHandler.pages = BROWSER_PAGES

    def test_soft_404_rendered(self):
        self.assertEqual(self.checker("free1")[0], "not_found")

    def test_http_404_rendered(self):
        self.assertEqual(self.checker("missing404")[0], "not_found")

    def test_js_rendered_profile_is_taken(self):
        # Text is written by JavaScript after 300 ms, so this checks the settle wait.
        self.assertEqual(self.checker("taken1")[0], "taken")

    def test_login_page_is_unknown(self):
        self.assertEqual(self.checker("login")[0], "unknown")


class _PageHandler(BaseHTTPRequestHandler):
    pages = {}

    def do_GET(self):
        status, body = self.pages.get(self.path, (404, "<html></html>"))
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
