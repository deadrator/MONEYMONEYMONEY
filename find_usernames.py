#!/usr/bin/env python3
"""
find_usernames.py - hunt for short (2-5 character) Instagram usernames that
look unclaimed.

Candidate sources (--style)
---------------------------
  dictionary    (default) real English words, 2-5 letters, from a lightweight
                offline source: a word must appear in BOTH the GCIDE and Webster
                dictionaries (via the `english-words` package) AND have a
                usable frequency score (via `wordfreq`). The frequency band
                (--min-zipf / --max-zipf) keeps words that are real but not
                overused, e.g. "lilt", "ferret", "plaid". No AI model, no
                network needed for word generation.
  wordlist      words from your own file (--wordlist), one per line.
  pronounceable made-up but pronounceable strings (FAH, BAX, ZOLI ...).
  exhaustive    every pronounceable CV/CVC string, shortest first.

How availability is checked
---------------------------
Instagram has no official public "is this username free?" API. The script
requests the public profile page and classifies the response:

  not_found     HTTP 404, or a 200 page saying "Sorry, this page isn't available".
                The only result reported as a possible hit.
  taken         a 200 page that is a real profile for this exact username.
  unknown       anything else (login redirect, unrecognised page). Never a hit.
  rate_limited  HTTP 429: the script backs off.
  blocked       HTTP 401/403: Instagram refused the request.
  error         network failure or other status.

A "not found" page does not prove the name is free. Instagram shows the same
page for deleted and banned accounts, so treat every hit as a *lead* and
confirm it in the Instagram app before relying on it.

Caveats
-------
* Instagram rate-limits aggressively. The script sleeps between requests and
  backs off on HTTP 429. Keep --delay high.
* Instagram's Terms of Use restrict automated access. Keep volume low and use
  this for your own research.
* Most dictionary words of 2-5 letters are already claimed, so expect to check
  a lot of candidates before getting hits.
* Dictionary sources are imperfect; a few odd tokens (abbreviations, names) can
  slip through. Review the hits yourself.
* Instagram rules are enforced: letters, numbers, periods and underscores only;
  no leading/trailing period; no double period.

Install (for the dictionary style)
----------------------------------
    pip install -r requirements.txt

Usage
-----
    python3 find_usernames.py                          # 5 dictionary-word hits
    python3 find_usernames.py --count 3 --max-len 4    # 3 hits, max 4 letters
    python3 find_usernames.py --min-zipf 3 --max-zipf 5   # more common words
    python3 find_usernames.py --dry-run --count 5      # list candidates only
    python3 find_usernames.py --style pronounceable    # made-up names instead
    python3 find_usernames.py --style wordlist --wordlist words.txt
    python3 find_usernames.py --output hits.txt

Python 3.8+. The dictionary style needs `wordfreq` and `english-words`; every
other style is standard library only.
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

PROFILE_URL = "https://www.instagram.com/{username}/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Instagram username rules: letters, digits, periods, underscores.
# No leading/trailing period, no consecutive periods, max 30 chars.
_VALID_RE = re.compile(r"^(?!\.)(?!.*\.\.)(?!.*\.$)[A-Za-z0-9._]{1,30}$")

CONSONANTS = "bcdfghjklmnprstvz"
VOWELS = "aeiou"

STYLES = ("dictionary", "wordlist", "pronounceable", "exhaustive")


def is_valid_username(name: str) -> bool:
    return bool(_VALID_RE.match(name))


# ---------------------------------------------------------------------------
# Dictionary words (lightweight, offline, no AI model)
# ---------------------------------------------------------------------------

def dictionary_words(
    min_len: int,
    max_len: int,
    min_zipf: float = 2.5,
    max_zipf: float = 4.5,
) -> List[str]:
    """
    Return lowercase real English words of min_len..max_len letters.

    A word qualifies if it is in both the GCIDE and Webster's dictionaries
    (drops most junk tokens) and its wordfreq Zipf score is within
    [min_zipf, max_zipf]. Zipf scores run roughly 0-8: ~3 is uncommon,
    ~5 is very common, <2 is obscure.
    """
    try:
        from english_words import get_english_words_set
        from wordfreq import zipf_frequency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The dictionary style needs extra packages. Run: "
            "pip install -r requirements.txt  (or use --style pronounceable)"
        ) from exc

    gcide = get_english_words_set(["gcide"], lower=True, alpha=True)
    webster = get_english_words_set(["web2"], lower=True, alpha=True)
    both = gcide & webster

    words = [
        w for w in both
        if min_len <= len(w) <= max_len
        and w.isalpha()
        and is_valid_username(w)
        and min_zipf <= zipf_frequency(w, "en") <= max_zipf
    ]
    return sorted(words)


def load_wordlist(path: str, min_len: int, max_len: int) -> List[str]:
    words = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            w = line.strip().lower()
            if min_len <= len(w) <= max_len and w.isalpha():
                words.append(w)
    # de-duplicate, keep order
    return list(dict.fromkeys(words))


# ---------------------------------------------------------------------------
# Made-up pronounceable strings
# ---------------------------------------------------------------------------

def pronounceable(length: int, rng: random.Random) -> Iterator[str]:
    """Yield random pronounceable strings of a given length (CV / CVC style)."""
    while True:
        start_consonant = rng.random() < 0.8
        chars = []
        for i in range(length):
            is_cons = (i % 2 == 0) == start_consonant
            chars.append(rng.choice(CONSONANTS if is_cons else VOWELS))
        yield "".join(chars)


def exhaustive_pronounceable(length: int) -> Iterator[str]:
    """Every alternating consonant/vowel string of a length (deterministic)."""
    for start_consonant in (True, False):
        pools = []
        for i in range(length):
            is_cons = (i % 2 == 0) == start_consonant
            pools.append(CONSONANTS if is_cons else VOWELS)
        for combo in itertools.product(*pools):
            yield "".join(combo)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    style: str,
    min_len: int,
    max_len: int,
    word_list: Optional[List[str]],
    rng: random.Random,
) -> Iterator[str]:
    """Yield candidate usernames for the chosen style, without duplicates."""
    seen = set()

    def fresh(items: Iterable[str]) -> Iterator[str]:
        for item in items:
            if item not in seen and is_valid_username(item):
                seen.add(item)
                yield item

    if style in ("dictionary", "wordlist"):
        shuffled = list(word_list or [])
        rng.shuffle(shuffled)
        yield from fresh(shuffled)
        return

    if style == "exhaustive":
        for length in range(min_len, max_len + 1):
            yield from fresh(exhaustive_pronounceable(length))
        return

    # pronounceable: random lengths in range, endless
    lengths = list(range(min_len, max_len + 1))
    generators = {n: pronounceable(n, rng) for n in lengths}
    while True:
        n = rng.choice(lengths)
        candidate = next(generators[n])
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------
#
# Labels returned by classify_response():
#   not_found    - Instagram says the profile does not exist. The ONLY label
#                  that counts as a possible hit. Still confirm in the app.
#   taken        - the page is a real profile for this exact username.
#   unknown      - anything we cannot classify (login wall, odd page, changed
#                  HTML). Never treated as available.
#   rate_limited - HTTP 429. The script backs off.
#   blocked      - HTTP 401/403. Instagram refused the request.
#   error        - network failure or other HTTP status.
#
# Why a 200 is not automatically "taken": Instagram can serve a "Sorry, this
# page isn't available" page with HTTP 200, and it shows that same page for
# deleted or banned accounts as well as unclaimed names. So the body is read.
# A 404 is treated as weak evidence of "not found", not proof of availability.
#
# The marker strings below are NOT verified against live Instagram HTML (the
# sandbox cannot reach instagram.com). If they do not match, results fall to
# "unknown" rather than "available", so the failure mode is safe. Adjust these
# after checking a real page in your own browser.

NOT_FOUND_MARKERS = (
    "sorry, this page isn't available",
    "sorry, this page isn\u2019t available",  # curly apostrophe variant
)
_OG_URL_RE = re.compile(
    r"""<meta[^>]+property=["']og:url["'][^>]+content=["']"""
    r"""https?://(?:www\.)?instagram\.com/([A-Za-z0-9._]+)/?["']""",
    re.IGNORECASE,
)
MAX_BODY_BYTES = 512 * 1024
DEFINITIVE = ("not_found", "taken")


def _owner_matches(username: str, body: str) -> bool:
    match = _OG_URL_RE.search(body)
    return bool(match) and match.group(1).lower() == username.lower()


def classify_response(
    username: str,
    status: int,
    location: Optional[str],
    body: str,
) -> Tuple[str, str]:
    """Pure function: turn an HTTP response into (label, reason). Testable offline."""
    if status == 404:
        return "not_found", "HTTP 404"
    if status == 429:
        return "rate_limited", "HTTP 429 (too many requests)"
    if status in (401, 403):
        return "blocked", f"HTTP {status} (Instagram refused the request)"
    if 300 <= status < 400:
        return "unknown", f"redirected to {location or '?'} (login wall or unclear)"
    if status != 200:
        return "error", f"HTTP {status}"

    lowered = body.lower()
    if any(marker in lowered for marker in NOT_FOUND_MARKERS):
        return "not_found", "200, but 'page isn't available' page"
    if _owner_matches(username, body):
        return "taken", "200, profile page for this username"
    return "unknown", "200, but neither a profile nor a not-found page"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stop urllib following redirects so a login redirect is visible."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def check_profile(username: str, timeout: float = 10.0) -> Tuple[str, str]:
    """Fetch the public profile page and classify it. Returns (label, reason)."""
    req = urllib.request.Request(
        PROFILE_URL.format(username=username),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            body = resp.read(MAX_BODY_BYTES).decode("utf-8", "replace")
            return classify_response(username, resp.status, None, body)
    except urllib.error.HTTPError as exc:
        # Redirects and error statuses both arrive here.
        location = exc.headers.get("Location") if exc.headers else None
        try:
            body = exc.read(MAX_BODY_BYTES).decode("utf-8", "replace")
        except OSError:
            body = ""
        return classify_response(username, exc.code, location, body)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "error", f"network error: {exc}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    count: int,
    min_len: int,
    max_len: int,
    style: str,
    word_list: Optional[List[str]],
    delay: float,
    max_checks: int,
    output: Optional[str],
    checker: Callable[[str], Tuple[str, str]] = check_profile,
    seed: Optional[int] = None,
    max_consecutive_unknown: int = 25,
) -> List[str]:
    rng = random.Random(seed)
    hits: List[str] = []
    counts: Dict[str, int] = {}
    checks = 0
    consecutive_bad = 0
    backoff = 60.0

    for candidate in generate_candidates(style, min_len, max_len, word_list, rng):
        if len(hits) >= count or checks >= max_checks:
            break

        label, reason = checker(candidate)
        checks += 1
        counts[label] = counts.get(label, 0) + 1

        if label == "rate_limited":
            print(f"  rate limited - backing off {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 900)
        else:
            backoff = 60.0

        if label in DEFINITIVE:
            consecutive_bad = 0
        else:
            consecutive_bad += 1

        if label == "not_found":
            hits.append(candidate)
            print(f"[+] @{candidate}  possibly AVAILABLE ({reason}) "
                  f"[{len(hits)}/{count}]")
            if output:
                with open(output, "a", encoding="utf-8") as fh:
                    fh.write(candidate + "\n")
        elif label == "taken":
            print(f"[-] @{candidate}  taken ({reason})", file=sys.stderr)
        else:
            print(f"[?] @{candidate}  {label}: {reason}", file=sys.stderr)

        if consecutive_bad >= max_consecutive_unknown:
            print(
                f"\nStopping: {consecutive_bad} results in a row were not definitive. "
                "Instagram is probably blocking or challenging these requests. "
                "Slow down (--delay) or check the page in a browser.",
                file=sys.stderr,
            )
            break

        if label not in ("rate_limited",) and delay > 0:
            time.sleep(delay)

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    print(
        f"\nChecked {checks} candidates ({summary}). "
        f"{len(hits)} possibly-available. Confirm each one in the Instagram app "
        "before using it.",
    )
    return hits


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find 2-5 character Instagram usernames that look unclaimed.",
    )
    p.add_argument("--count", type=int, default=5,
                   help="how many available usernames to find (1-5, default 5)")
    p.add_argument("--min-len", type=int, default=2, help="min length (default 2)")
    p.add_argument("--max-len", type=int, default=5, help="max length (default 5)")
    p.add_argument("--style", choices=STYLES, default="dictionary",
                   help="candidate source (default: dictionary)")
    p.add_argument("--wordlist", help="file of words, one per line (for --style wordlist)")
    p.add_argument("--min-zipf", type=float, default=2.5,
                   help="dictionary style: min word frequency, Zipf scale (default 2.5)")
    p.add_argument("--max-zipf", type=float, default=4.5,
                   help="dictionary style: max word frequency, Zipf scale (default 4.5)")
    p.add_argument("--delay", type=float, default=5.0,
                   help="seconds between checks (default 5; keep it high)")
    p.add_argument("--max-checks", type=int, default=300,
                   help="stop after this many lookups (default 300)")
    p.add_argument("--output", help="append hits to this file")
    p.add_argument("--seed", type=int, help="random seed for reproducible runs")
    p.add_argument("--dry-run", action="store_true",
                   help="print candidates without checking Instagram")
    args = p.parse_args(argv)

    if not 1 <= args.count <= 5:
        p.error("--count must be between 1 and 5")
    if not 1 <= args.min_len <= args.max_len <= 30:
        p.error("lengths must satisfy 1 <= min-len <= max-len <= 30")
    if args.style == "wordlist" and not args.wordlist:
        p.error("--style wordlist requires --wordlist FILE")
    if args.min_zipf > args.max_zipf:
        p.error("--min-zipf must be <= --max-zipf")
    return args


def load_words_for(args: argparse.Namespace) -> Optional[List[str]]:
    if args.style == "dictionary":
        return dictionary_words(args.min_len, args.max_len,
                                args.min_zipf, args.max_zipf)
    if args.style == "wordlist":
        return load_wordlist(args.wordlist, args.min_len, args.max_len)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        word_list = load_words_for(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if word_list is not None:
        print(f"Loaded {len(word_list)} candidate words (style: {args.style})")
        if not word_list:
            print("No words matched. Widen --min-zipf/--max-zipf or the length range.",
                  file=sys.stderr)
            return 1

    if args.dry_run:
        rng = random.Random(args.seed)
        gen = generate_candidates(args.style, args.min_len, args.max_len,
                                  word_list, rng)
        for _, cand in zip(range(args.count * 4), gen):
            print(f"@{cand}")
        return 0

    try:
        hits = run(
            count=args.count,
            min_len=args.min_len,
            max_len=args.max_len,
            style=args.style,
            word_list=word_list,
            delay=args.delay,
            max_checks=args.max_checks,
            output=args.output,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130

    if not hits:
        print("No available candidates found. Try --max-checks higher or another --style.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
