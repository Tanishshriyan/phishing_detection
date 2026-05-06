"""Feature extraction utilities for phishing URL detection."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_KEYWORDS = {
    "account",
    "bank",
    "billing",
    "confirm",
    "free",
    "gift",
    "invoice",
    "login",
    "malware",
    "password",
    "pay",
    "phishing",
    "secure",
    "signin",
    "scam",
    "support",
    "unlock",
    "update",
    "verify",
    "wallet",
}

THREAT_KEYWORDS = {
    "fraud",
    "malware",
    "phishing",
    "scam",
}

GENERIC_HOSTING_DOMAINS = {
    "firebaseapp.com",
    "fly.dev",
    "github.io",
    "glitch.me",
    "herokuapp.com",
    "netlify.app",
    "onrender.com",
    "pages.dev",
    "railway.app",
    "replit.app",
    "surge.sh",
    "vercel.app",
    "web.app",
}

SUSPICIOUS_TLDS = {
    "biz",
    "cc",
    "cf",
    "click",
    "country",
    "ga",
    "gq",
    "info",
    "link",
    "live",
    "loan",
    "ml",
    "monster",
    "online",
    "pw",
    "rest",
    "ru",
    "shop",
    "tk",
    "top",
    "work",
    "xyz",
}

URL_SHORTENERS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "shorturl.at",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
}


def _normalize_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return "http://invalid.local"
    if "://" not in value:
        return f"http://{value}"
    return value


def _is_ip_address(hostname: str) -> int:
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def _matches_domain(hostname: str, domains: set[str]) -> int:
    lowered = hostname.lower()
    return int(any(lowered == domain or lowered.endswith(f".{domain}") for domain in domains))


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    frequencies = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in frequencies.values())


def extract_url_features(url: str) -> dict[str, float]:
    """Extracts numeric features from a URL string."""
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    lowered = normalized.lower()
    hostname = parsed.hostname or ""
    hostname_lowered = hostname.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""

    token_candidates = [token for token in re.split(r"[\W_]+", lowered) if token]
    host_tokens = [token for token in re.split(r"[\W_]+", hostname_lowered) if token]
    digit_count = sum(char.isdigit() for char in normalized)
    letter_count = sum(char.isalpha() for char in normalized)
    special_count = sum(not char.isalnum() for char in normalized)
    suspicious_hits = sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS)
    host_suspicious_hits = sum(token in SUSPICIOUS_KEYWORDS for token in host_tokens)
    host_threat_hits = sum(token in THREAT_KEYWORDS for token in host_tokens)
    subdomain_count = max(hostname.count(".") - 1, 0)
    query_param_count = query.count("&") + 1 if query else 0

    try:
        has_port = 1 if parsed.port is not None else 0
    except ValueError:
        has_port = 0

    features = {
        "url_length": float(len(normalized)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "count_dots": float(normalized.count(".")),
        "count_hyphen": float(normalized.count("-")),
        "count_at": float(normalized.count("@")),
        "count_question": float(normalized.count("?")),
        "count_equal": float(normalized.count("=")),
        "count_ampersand": float(normalized.count("&")),
        "count_percent": float(normalized.count("%")),
        "count_slash": float(normalized.count("/")),
        "count_digits": float(digit_count),
        "count_letters": float(letter_count),
        "count_special_chars": float(special_count),
        "digit_ratio": float(digit_count / max(len(normalized), 1)),
        "special_ratio": float(special_count / max(len(normalized), 1)),
        "num_subdomains": float(subdomain_count),
        "uses_https": float(parsed.scheme.lower() == "https"),
        "has_ip_address": float(_is_ip_address(hostname)),
        "has_suspicious_tld": float(tld in SUSPICIOUS_TLDS),
        "has_generic_hosting_domain": float(_matches_domain(hostname, GENERIC_HOSTING_DOMAINS)),
        "has_shortener": float(hostname in URL_SHORTENERS),
        "suspicious_keyword_hits": float(suspicious_hits),
        "host_suspicious_keyword_hits": float(host_suspicious_hits),
        "host_threat_keyword_hits": float(host_threat_hits),
        "contains_login_hint": float(any(word in lowered for word in ("login", "signin", "verify"))),
        "contains_security_hint": float(any(word in lowered for word in ("secure", "update", "confirm"))),
        "entropy": float(_shannon_entropy(normalized)),
        "token_count": float(len(token_candidates)),
        "avg_token_length": float(sum(len(token) for token in token_candidates) / max(len(token_candidates), 1)),
        "longest_token_length": float(max((len(token) for token in token_candidates), default=0)),
        "path_depth": float(path.count("/")),
        "query_param_count": float(query_param_count),
        "port_present": float(has_port),
    }
    return features


def extract_features_dataframe(urls: Iterable[str]) -> pd.DataFrame:
    """Builds a DataFrame of extracted URL features."""
    records = [extract_url_features(url) for url in urls]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)
