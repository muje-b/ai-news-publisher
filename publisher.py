#!/usr/bin/env python3
"""
AI News Auto-Publisher v2 — NVIDIA NIM Edition
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

# ============================================================
# CONFIGURATION
# ============================================================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BLOG_MODEL = os.environ.get("BLOG_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
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
        "type": "news"
    },
    "google_news_ai": {
        "url": "https://news.google.com/rss/search?q=artificial+intelligence+new+model+OR+new+tool+OR+release+OR+launch&hl=en-US&gl=US&ceid=US:en",
        "category": "News",
        "type": "news"
    },
}

# ============================================================
# STATE
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"processed": [], "last_run": None, "stats": {"total_posts": 0, "total_tweets": 0}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ============================================================
# HELPERS
# ============================================================
def get_hash(identifier):
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def clean_html(text):
    return re.sub(r'<[^>]+>', '', text).strip()


def safe_filename(text, max_len=60):
    name = re.sub(r'[^\w\s-]', '', text).strip().lower()
    name = re.sub(r'[-\s]+', '-', name)
    return name[:max_len]

# ============================================================
# NVIDIA NIM API — OpenAI-compatible, free tier
# ============================================================
def call_nvidia(prompt, model=None, max_tokens=2048):
    if model is None:
        model = BLOG_MODEL

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"  ⚠ NVIDIA error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  ⚠ NVIDIA exception: {e}")
        return None

# ============================================================
# DATA FETCHING
# ============================================================
def fetch_rss_feeds():
    items = []
    for source_id, source_info in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(source_info["url"])
            for entry in feed.entries[:10]:
                summary = clean_html(entry.get("summary", entry.get("description", "")))
                items.append({
                    "title": entry.get("title", "Untitled"),
                    "link": entry.get("link", ""),
                    "summary": summary[:800],
                    "published": entry.get("published", ""),
                    "source": source_id,
                    "category": source_info["category"],
                    "type": source_info["type"],
                    "hash": get_hash(entry.get("link", entry.get("id", ""))),
                })
        except Exception as e:
            print(f"  ⚠ RSS error ({source_id}): {e}")
    return items


def fetch_huggingface_models():
    items = []
    try:
        response = requests.get(
            "https://huggingface.co/api/models",
            params={"sort": "created", "direction": "-1", "limit": 5, "pipeline_tag": "text-generation"},
            timeout=15
        )
        if response.status_code == 200:
            for model in response.json():
                tags = model.get("tags", [])
                items.append({
                    "title": f"New Model: {model.get('modelId', 'Unknown')}",
                    "link": f"https://huggingface.co/{model.get('modelId', '')}",
                    "summary": f"Author: {model.get('author', 'Unknown')}. Pipeline: text-generation. Tags: {', '.join(tags[:6])}. Downloads: {model.get('downloads', 0):,}. Likes: {model.get('likes', 0)}.",
                    "published": model.get("createdAt", ""),
                    "source": "huggingface",
                    "category": "Models",
                    "type": "model",
                    "hash": get_hash(f"hf-{model.get('modelId', '')}"),
                })
    except Exception as e:
        print(f"  ⚠ HuggingFace error: {e}")
    return items


def fetch_github_trending():
    items = []
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={
                "q": f"created:>{since} (machine-learning OR llm OR ai-agent OR gpt OR transformer)",
                "sort": "stars", "order": "desc", "per_page": 5
            },
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=15
        )
        if response.status_code == 200:
            for repo in response.json().get("items", []):
                desc = repo.get("description", "") or "No description"
                items.append({
                    "title": f"New Open Source Tool: {repo.get('name', 'Unknown')}",
                    "link": repo.get("html_url", ""),
                    "summary": f"{desc}. Stars: {repo.get('stargazers_count', 0):,}. Language: {repo.get('language', 'N/A')}. Forks: {repo.get('forks_count', 0):,}.",
                    "published": repo.get("created_at", ""),
                    "source": "github",
                    "category": "Tools",
                    "type": "tool",
                    "hash": get_hash(f"gh-{repo.get('full_name', '')}"),
                })
    except Exception as e:
        print(f"  ⚠ GitHub error: {e}")
    return items


def fetch_all_sources():
    print("📡 Fetching from 8 sources...")
    all_items = []
    all_items.extend(fetch_rss_feeds())
    all_items.extend(fetch_huggingface_models())
    all_items.extend(fetch_github_trending())
    print(f"  ✅ Fetched {len(all_items)} total items")
    return all_items

# ============================================================
# QUALITY SCORING — filters junk before wasting API calls
# ============================================================
def score_item(item):
    """Use fast 8B model to score newsworthiness 1-10."""
    prompt = f"""Rate this AI news item's newsworthiness 1-10.
Give 8+ ONLY if: genuinely new (last 48h), names specific model/tool/company, interesting to AI developers or researchers.
Give 3 or below if: vague, clickbait, duplicate, old news, not about AI.

Title: {item['title']}
Info: {item['summary'][:300]}

Respond with ONLY a single number 1-10. Nothing else."""

    result = call_nvidia(prompt, model=TWEET_MODEL, max_tokens=5)
    try:
        score = int(re.search(r'\d+', result).group())
        return min(max(score, 1), 10)
    except:
        return 5


def filter_new_items(items, state):
    processed = set(state.get("processed", []))
    candidates = [item for item in items if item["hash"] not in processed]
    
    if not candidates:
        return []
    
    print(f"  🔍 Scoring {min(len(candidates), 15)} candidates for quality...")
    scored = []
    for item in candidates[:15]:
        s = score_item(item)
        scored.append((s, item))
        print(f"    [{s}/10] {item['title'][:55]}...")
        time.sleep(1)
    
    scored.sort(key=lambda x: x[0], reverse=True)
    high_quality = [(s, item) for s, item in scored if s >= MIN_QUALITY_SCORE]
    
    print(f"  ✅ {len(high_quality)} items passed quality filter (score >= {MIN_QUALITY_SCORE})")
    return [item for _, item in high_quality[:MAX_POSTS_PER_RUN]]

# ============================================================
# CONTENT GENERATION
# ============================================================
def generate_blog_post(item):
    prompt = f"""You are a tech writer for a popular AI news website. Write an engaging blog post about this {item['type']}.

TITLE: {item['title']}
SOURCE: {item['source'].upper()}
CATEGORY: {item['category']}
RAW INFO: {item['summary']}

Follow this EXACT structure using HTML tags:
1. Opening paragraph — hook the reader, tell them what happened and why they should care
2. <h2>What You Need to Know</h2> — 2-3 paragraphs explaining key details in simple, clear language
3. <h2>Why It Matters</h2> — 1-2 paragraphs on significance for AI developers, researchers, or users
4. <h2>Key Details</h2> — bulleted list (<ul><li>) of 4-6 most important facts
5. <h2>What's Next</h2> — 1 paragraph on what to expect

STRICT RULES:
- 400-600 words total
- Do NOT include the title or any <h1> tag
- Do NOT include <html>, <head>, or <body> tags
- Use ONLY: <h2>, <p>, <ul>, <li>, <strong>, <em>, <a> tags
- Write conversationally but professionally
- NEVER use: revolutionary, game-changing, paradigm shift, groundbreaking, unprecedented
- Be specific with numbers, names, dates — not vague
- If raw info is sparse, add reasonable context about the AI domain"""

    return call_nvidia(prompt, model=BLOG_MODEL, max_tokens=2048)


def generate_tweet(item, post_url=""):
    link_space = len(post_url) + 1 if post_url else 0
    max_chars = 270 - link_space

    prompt = f"""Write ONE tweet about this AI {item['type']}. Maximum {max_chars} characters.

TITLE: {item['title']}
KEY INFO: {item['summary'][:200]}

RULES:
- Start with a hook, not "Breaking" or "Exciting"
- Be specific (name the model/tool/company)
- Include 1-2 relevant hashtags
- Make someone want to click the link
- Output ONLY the tweet text, nothing else, no quotes"""

    tweet = call_nvidia(prompt, model=TWEET_MODEL, max_tokens=80)
    if tweet:
        tweet = tweet.strip().strip('"').strip("'").strip()
        if post_url:
            tweet = f"{tweet}\n{post_url}"
        if len(tweet) > 280:
            if "\n" in tweet:
                lines = tweet.split("\n")
                tweet = lines[0][:240]
                if post_url:
                    tweet += f"\n{post_url}"
            else:
                tweet = tweet[:240] + "..."
    return tweet

# ============================================================
# IMAGE GENERATION — Free via Pollinations.ai
# ============================================================
def generate_and_upload_image(item, post_id):
    """Generate a header image and set it as WordPress featured image."""
    if not all([WP_URL, WP_USERNAME, WP_APP_PASSWORD, post_id]):
        return None

    prompt = f"""Write an image generation prompt (under 80 words) for a blog header image about:
{item['title']}
{item['summary'][:200]}
Style: modern tech, dark blue purple gradient, abstract AI visualization, no text, clean, professional.
Output ONLY the image prompt."""

    image_prompt = call_nvidia(prompt, model=TWEET_MODEL, max_tokens=100)
    if not image_prompt:
        return None

    try:
        encoded = requests.utils.quote(image_prompt.strip())
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1200&height=630&nologo=true&seed={item['hash']}"
        
        img_response = requests.get(img_url, timeout=45)
        if img_response.status_code == 200 and len(img_response.content) > 5000:
            media_response = requests.post(
                f"{WP_URL}/wp-json/wp/v2/media",
                headers={
                    "Content-Disposition": 'attachment; filename="header.jpg"',
                    "Content-Type": "image/jpeg",
                },
                data=img_response.content,
                auth=(WP_USERNAME, WP_APP_PASSWORD),
                timeout=20
            )
            if media_response.status_code in [200, 201]:
                media_id = media_response.json().get("id")
                requests.post(
                    f"{WP_URL}/wp-json/wp/v2/posts/{post_id}",
                    json={"featured_media": media_id},
                    auth=(WP_USERNAME, WP_APP_PASSWORD),
                    timeout=10
                )
                print(f"  🖼️ Featured image added")
                return True
    except Exception as e:
        print(f"  ⚠ Image failed: {e}")
    return None

# ============================================================
# PUBLISHING — WordPress
# ============================================================
def publish_to_wordpress(title, content_html, item):
    has_wp = all([WP_URL, WP_USERNAME, WP_APP_PASSWORD])

    full_content = f"""{content_html}
<blockquote>📌 <strong>Source:</strong> <a href="{item.get('link', '')}" target="_blank" rel="noopener noreferrer">{item.get('source', '').replace('_', ' ').title()}</a></blockquote>"""

    if has_wp:
        data = {
            "title": title,
            "content": full_content,
            "status": "publish",
            "slug": safe_filename(title),
        }
        try:
            response = requests.post(
                f"{WP_URL}/wp-json/wp/v2/posts",
                json=data,
                auth=(WP_USERNAME, WP_APP_PASSWORD),
                timeout=20
            )
            if response.status_code in [200, 201]:
                post_data = response.json()
                post_url = post_data.get("link", "")
                post_id = post_data.get("id")
                print(f"  ✅ Published: {post_url}")
                
                # Add featured image
                generate_and_upload_image(item, post_id)
                
                return post_url
            else:
                print(f"  ⚠ WordPress error {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"  ⚠ WordPress exception: {e}")

    os.makedirs(DRAFTS_DIR, exist_ok=True)
    filename = f"{DRAFTS_DIR}/{safe_filename(title)}.html"
    with open(filename, "w") as f:
        f.write(f"<h1>{title}</h1>\n{full_content}")
    print(f"  📝 Draft saved: {filename}")
    return None

# ============================================================
# PUBLISHING — X/Twitter
# ============================================================
def post_to_x(tweet_text):
    has_x = all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET])

    if has_x:
        try:
            auth = OAuth1(
                X_API_KEY, client_secret=X_API_SECRET,
                resource_owner_key=X_ACCESS_TOKEN,
                resource_owner_secret=X_ACCESS_TOKEN_SECRET,
            )
            response = requests.post(
                "https://api.twitter.com/2/tweets",
                json={"text": tweet_text},
                auth=auth, timeout=15
            )
            if response.status_code == 201:
                print(f"  ✅ Tweeted: {tweet_text[:60]}...")
                return True
            else:
                print(f"  ⚠ X error {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"  ⚠ X exception: {e}")

    os.makedirs(TWEETS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"{TWEETS_DIR}/tweet_{ts}.txt", "w") as f:
        f.write(tweet_text)
    print(f"  📝 Tweet saved: {TWEETS_DIR}/tweet_{ts}.txt")
    return False

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print(f"🤖 AI News Auto-Publisher v2 (NVIDIA NIM)")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    if not NVIDIA_API_KEY:
        print("❌ NVIDIA_API_KEY not set. Add it in GitHub Secrets.")
        return

    wp_ok = all([WP_URL, WP_USERNAME, WP_APP_PASSWORD])
    x_ok = all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET])
    print(f"🔑 NVIDIA: ✅ {BLOG_MODEL}")
    print(f"📡 WordPress: {'✅ Connected' if wp_ok else '⚠️ Saving drafts'}")
    print(f"🐦 X/Twitter: {'✅ Connected' if x_ok else '⚠️ Saving files'}")
    print(f"🎯 Min quality score: {MIN_QUALITY_SCORE}/10")
    print()

    state = load_state()

    all_items = fetch_all_sources()
    new_items = filter_new_items(all_items, state)

    if not new_items:
        print("📭 No new high-quality items. Everything up to date.")
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        return

    print(f"\n📝 Processing {len(new_items)} items...\n")

    posts_published = 0
    tweets_sent = 0

    for i, item in enumerate(new_items, 1):
        print(f"{'─' * 55}")
        print(f"[{i}/{len(new_items)}] {item['title'][:65]}")

        print("  🔄 Generating blog post...")
        content = generate_blog_post(item)
        if not content:
            print("  ⏭ Skipped (generation failed)")
            continue

        print("  🔄 Publishing...")
        post_url = publish_to_wordpress(item["title"], content, item)

        if post_url:
            posts_published += 1
            print("  🔄 Generating & posting tweet...")
            tweet = generate_tweet(item, post_url)
            if tweet and post_to_x(tweet):
                tweets_sent += 1
        else:
            tweet = generate_tweet(item)
            if tweet:
                post_to_x(tweet)

        state["processed"].append(item["hash"])
        if i < len(new_items):
            time.sleep(5)

    state["processed"] = state["processed"][-1000:]
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["stats"]["total_posts"] = state["stats"].get("total_posts", 0) + posts_published
    state["stats"]["total_tweets"] = state["stats"].get("total_tweets", 0) + tweets_sent
    save_state(state)

    print(f"\n{'=' * 55}")
    print(f"✅ Done! {posts_published} posts, {tweets_sent} tweets")
    print(f"   Lifetime: {state['stats']['total_posts']} posts, {state['stats']['total_tweets']} tweets")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
