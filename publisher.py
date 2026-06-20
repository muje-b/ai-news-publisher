#!/usr/bin/env python3
"""
AI News Auto-Publisher v2 — NVIDIA NIM Edition (OpenAI SDK + Nemotron Thinking)
Free. Automated. Open-source models.
"""

import feedparser
import json
import os
import requests
import hashlib
import time
import re
import base64
from datetime import datetime, timedelta, timezone
from requests_oauthlib import OAuth1
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Using the Nemotron "thinking" model for high-quality blog posts
BLOG_MODEL = os.environ.get("BLOG_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# Using the fast 8B model for tweets and quick scoring
TWEET_MODEL = os.environ.get("TWEET_MODEL", "meta/llama-3.1-8b-instruct")

WP_URL = os.environ.get("WP_URL", "").rstrip("/")
WP_USERNAME = os.environ.get("WP_USERNAME", "")
WP_APP_PASSWORD = os.environ.get("WP_APP_PASSWORD", "")

X_API_KEY = os.environ.get("X_API_KEY", "")
X_API_SECRET = os.environ.get("X_API_SECRET", "")
X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET", "")

MAX_POSTS_PER_RUN = int(os.environ.get("MAX_POSTS_PER_RUN", "3"))
MIN_QUALITY_SCORE = int(os.environ.get("MIN_QUALITY_SCORE", "7"))
STATE_FILE = "state.json"
DRAFTS_DIR = "drafts"
TWEETS_DIR = "tweets"

# ============================================================
# DATA SOURCES
# ============================================================
RSS_FEEDS = {
    "arxiv_ai": {
        "url": "http://arxiv.org/rss/cs.AI",
        "category": "Papers",
        "type": "paper"
    },
    "arxiv_ml": {
        "url": "http://arxiv.org/rss/cs.LG",
        "category": "Papers",
        "type": "paper"
    },
    "techcrunch_ai": {
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "News",
        "type": "news"
    },
    "verge_ai": {
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "category": "News",
        "
