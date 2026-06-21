#!/usr/bin/env python3
"""
AI News Auto-Publisher v2 — NVIDIA NIM Edition (Pure Requests - GitHub Safe)
"""

import feedparser
import json
import os
import requests
import hashlib
import time
import re
from datetime import datetime, timedelta, timezone
from requests_oauthlib import OAuth1

# ============================================================
# CONFIGURATION
# ============================================================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BLOG_MODEL = os.environ.get("BLOG_MODEL", "nvidia/nemotron-3-super-120b-a12b")
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
    "arxiv_ai": {"url": "http://arxiv.org/rss/cs.AI", "category": "Papers", "type": "paper"},
    "arxiv_ml": {"url": "http://arxiv.org/rss/cs.LG", "category": "Papers", "type": "paper"},
    "techcrunch_ai": {"url": "https://techcrunch.com/category/artificial-intelligence/feed/", "category": "News", "type": "news"},
    "verge_ai": {"url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "category": "News", "type": "news"},
    "google_news_ai": {"url": "https://news.google.com/rss/search?q=artificial+intelligence+new+model+OR+new+tool+OR+release+OR+launch&hl=en-US&gl=US&ceid=US:en", "category": "News", "type": "news"},
}

# ============================================================
# STATE & HELPERS
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"processed": [], "last_run": None, "stats": {"total_posts": 0, "total_tweets": 0}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_hash(id): return hashlib.sha256(id.encode()).hexdigest()[:16]
def clean_html(t): return re.sub(r'<[^>]+>', '', t).strip()
def safe_filename(t, m=60):
    n = re.sub(r'[^\w\s-]', '', t).strip().lower()
    return re.sub(r'[-\s]+', '-', n)[:m]

# ============================================================
# NVIDIA NIM API — Pure Requests (No OpenAI SDK crashes)
# ============================================================
def call_nvidia(prompt, model=None, max_tokens=2048):
    if model is None: model = BLOG_MODEL
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False
    }

    # Enable Nemotron "Thinking" mode if using the super model
    if "nemotron" in model and "super" in model:
        payload["temperature"] = 1
        payload["top_p"] = 0.95
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["reasoning_budget"] = max_tokens

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120, # Nemotron thinking takes a bit longer
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"  ⚠ NVIDIA error {response.status_code}: {response.text[:150]}")
            return None
    except Exception as e:
        print(f"  ⚠ NVIDIA exception: {e}")
        return None

# ============================================================
# DATA FETCHING
# ============================================================
def fetch_rss_feeds():
    items = []
    for sid, info in RSS_FEEDS.items():
        try:
            for e in feedparser.parse(info["url"]).entries[:10]:
                items.append({
                    "title": e.get("title", "Untitled"), "link": e.get("link", ""),
                    "summary": clean_html(e.get("summary", ""))[:800], "published": e.get("published", ""),
                    "source": sid, "category": info["category"], "type": info["type"],
                    "hash": get_hash(e.get("link", e.get("id", "")))
                })
        except Exception as e: print(f"  ⚠ RSS error ({sid}): {e}")
    return items

def fetch_huggingface_models():
    items = []
    try:
        for m in requests.get("https://huggingface.co/api/models", params={"sort": "created", "direction": "-1", "limit": 5, "pipeline_tag": "text-generation"}, timeout=15).json():
            items.append({
                "title": f"New Model: {m.get('modelId', '')}", "link": f"https://huggingface.co/{m.get('modelId', '')}",
                "summary": f"Author: {m.get('author', '')}. Tags: {', '.join(m.get('tags', [])[:6])}. Downloads: {m.get('downloads', 0):,}.",
                "published": m.get("createdAt", ""), "source": "huggingface", "category": "Models", "type": "model",
                "hash": get_hash(f"hf-{m.get('modelId', '')}")
            })
    except Exception as e: print(f"  ⚠ HuggingFace error: {e}")
    return items

def fetch_github_trending():
    items = []
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        for r in requests.get("https://api.github.com/search/repositories", params={"q": f"created:>{since} (machine-learning OR llm OR ai-agent)", "sort": "stars", "order": "desc", "per_page": 5}, headers={"Accept": "application/vnd.github.v3+json"}, timeout=15).json().get("items", []):
            items.append({
                "title": f"New Tool: {r.get('name', '')}", "link": r.get("html_url", ""),
                "summary": f"{r.get('description', '')}. ⭐ {r.get('stargazers_count', 0):,}. Lang: {r.get('language', 'N/A')}.",
                "published": r.get("created_at", ""), "source": "github", "category": "Tools", "type": "tool",
                "hash": get_hash(f"gh-{r.get('full_name', '')}")
            })
    except Exception as e: print(f"  ⚠ GitHub error: {e}")
    return items

def fetch_all_sources():
    print("📡 Fetching from 8 sources...")
    items = fetch_rss_feeds() + fetch_huggingface_models() + fetch_github_trending()
    print(f"  ✅ Fetched {len(items)} total items")
    return items

# ============================================================
# QUALITY SCORING
# ============================================================
def score_item(item):
    prompt = f"""Rate this AI news newsworthiness 1-10. 8+ ONLY if specific/new/important. 3 or below if vague/old.
Title: {item['title']}
Info: {item['summary'][:300]}
Respond with ONLY a number 1-10."""
    result = call_nvidia(prompt, model=TWEET_MODEL, max_tokens=5)
    try: return min(max(int(re.search(r'\d+', result).group()), 1), 10)
    except: return 5

def filter_new_items(items, state):
    candidates = [i for i in items if i["hash"] not in set(state.get("processed", []))]
    if not candidates: return []
    print(f"  🔍 Scoring {min(len(candidates), 15)} candidates...")
    scored = []
    for item in candidates[:15]:
        s = score_item(item)
        scored.append((s, item))
        print(f"    [{s}/10] {item['title'][:55]}...")
        time.sleep(1)
    scored.sort(key=lambda x: x[0], reverse=True)
    best = [i for s, i in scored if s >= MIN_QUALITY_SCORE]
    print(f"  ✅ {len(best)} items passed quality filter")
    return best[:MAX_POSTS_PER_RUN]

# ============================================================
# CONTENT GENERATION
# ============================================================
def generate_blog_post(item):
    prompt = f"""Write a blog post about this {item['type']}.
TITLE: {item['title']}
RAW INFO: {item['summary']}

Structure using HTML:
1. Hook paragraph
2. <h2>What You Need to Know</h2> (2-3 paragraphs)
3. <h2>Why It Matters</h2> (1-2 paragraphs)
4. <h2>Key Details</h2> (<ul><li> 4-6 bullets)
5. <h2>What's Next</h2> (1 paragraph)

RULES: 400-600 words. NO <h1> or <body> tags. ONLY use <h2>, <p>, <ul>, <li>, <strong>. NO buzzwords (revolutionary etc). Be specific."""
    return call_nvidia(prompt, model=BLOG_MODEL, max_tokens=2048)

def generate_tweet(item, post_url=""):
    mc = 270 - (len(post_url) + 1 if post_url else 0)
    prompt = f"""Write ONE tweet (max {mc} chars) about: {item['title']}. {item['summary'][:150]}
Rules: Specific hook, 1-2 hashtags, NO "Breaking". Output ONLY tweet text."""
    tweet = call_nvidia(prompt, model=TWEET_MODEL, max_tokens=80)
    if tweet:
        tweet = tweet.strip().strip('"').strip("'")
        if post_url: tweet = f"{tweet}\n{post_url}"
        if len(tweet) > 280: tweet = tweet[:240] + "..."
    return tweet

# ============================================================
# IMAGE GENERATION
# ============================================================
def generate_and_upload_image(item, post_id):
    if not all([WP_URL, WP_USERNAME, WP_APP_PASSWORD, post_id]): return None
    prompt = f"Image prompt for tech blog header about: {item['title']}. Modern, dark blue purple AI abstract, NO TEXT."
    img_prompt = call_nvidia(prompt, model=TWEET_MODEL, max_tokens=80)
    if not img_prompt: return None
    try:
        img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(img_prompt.strip())}?width=1200&height=630&nologo=true&seed={item['hash']}"
        img_res = requests.get(img_url, timeout=45)
        if img_res.status_code == 200 and len(img_res.content) > 5000:
            med_res = requests.post(f"{WP_URL}/wp-json/wp/v2/media", headers={"Content-Disposition": 'attachment; filename="header.jpg"', "Content-Type": "image/jpeg"}, data=img_res.content, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=20)
            if med_res.status_code in [200, 201]:
                requests.post(f"{WP_URL}/wp-json/wp/v2/posts/{post_id}", json={"featured_media": med_res.json().get("id")}, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=10)
                print(f"  🖼️ Featured image added")
    except Exception as e: print(f"  ⚠ Image failed: {e}")

# ============================================================
# PUBLISHING
# ============================================================
def publish_to_wordpress(title, content, item):
    full = f"""{content}
<blockquote>📌 <strong>Source:</strong> <a href="{item.get('link', '')}" target="_blank">{item.get('source', '').replace('_', ' ').title()}</a></blockquote>"""
    if all([WP_URL, WP_USERNAME, WP_APP_PASSWORD]):
        try:
            res = requests.post(f"{WP_URL}/wp-json/wp/v2/posts", json={"title": title, "content": full, "status": "publish", "slug": safe_filename(title)}, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=30)
            if res.status_code in [200, 201]:
                d = res.json()
                print(f"  ✅ Published: {d.get('link', '')}")
                generate_and_upload_image(item, d.get("id"))
                return d.get("link")
            else: print(f"  ⚠ WP error {res.status_code}: {res.text[:150]}")
        except Exception as e: print(f"  ⚠ WP exception: {e}")
    
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    with open(f"{DRAFTS_DIR}/{safe_filename(title)}.html", "w") as f: f.write(f"<h1>{title}</h1>{full}")
    print(f"  📝 Draft saved")
    return None

def post_to_x(text):
    if all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET]):
        try:
            auth = OAuth1(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)
            res = requests.post("https://api.twitter.com/2/tweets", json={"text": text}, auth=auth, timeout=15)
            if res.status_code == 201: print(f"  ✅ Tweeted: {text[:60]}..."); return True
            else: print(f"  ⚠ X error {res.status_code}")
        except Exception as e: print(f"  ⚠ X exception: {e}")
    os.makedirs(TWEETS_DIR, exist_ok=True)
    with open(f"{TWEETS_DIR}/tweet_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt", "w") as f: f.write(text)
    return False

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print(f"🤖 AI News Auto-Publisher (Nemotron Thinking)")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    if not NVIDIA_API_KEY:
        print("❌ NVIDIA_API_KEY not set.")
        return

    wp_ok = all([WP_URL, WP_USERNAME, WP_APP_PASSWORD])
    x_ok = all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET])
    print(f"🧠 Blog: {BLOG_MODEL}\n⚡ Tweet: {TWEET_MODEL}")
    print(f"📡 WP: {'✅' if wp_ok else '⚠️'} | 🐦 X: {'✅' if x_ok else '⚠️'}\n")

    state = load_state()
    new_items = filter_new_items(fetch_all_sources(), state)

    if not new_items:
        print("📭 No new high-quality items.")
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        return

    print(f"\n📝 Processing {len(new_items)} items...\n")
    pp, tt = 0, 0

    for i, item in enumerate(new_items, 1):
        print(f"{'─'*55}\n[{i}/{len(new_items)}] {item['title'][:65]}")
        print("  🔄 Generating blog post (Nemotron thinking)...")
        content = generate_blog_post(item)
        if not content: continue

        print("  🔄 Publishing...")
        url = publish_to_wordpress(item["title"], content, item)
        
        if url:
            pp += 1
            print("  🔄 Tweeting...")
            tweet = generate_tweet(item, url)
            if tweet and post_to_x(tweet): tt += 1
        else:
            tweet = generate_tweet(item)
            if tweet: post_to_x(tweet)

        state["processed"].append(item["hash"])
        if i < len(new_items): time.sleep(5)

    state["processed"] = state["processed"][-1000:]
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["stats"]["total_posts"] = state["stats"].get("total_posts", 0) + pp
    state["stats"]["total_tweets"] = state["stats"].get("total_tweets", 0) + tt
    save_state(state)

    print(f"\n{'='*55}\n✅ Done! {pp} posts, {tt} tweets\n{'='*55}")

if __name__ == "__main__":
    main()
