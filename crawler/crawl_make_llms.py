# crawler/crawl_make_llms.py
import asyncio, json, hashlib, textwrap, pathlib
from datetime import datetime
from urllib.parse import urldefrag

from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig,
    CacheMode, MemoryAdaptiveDispatcher
)

# -------------------------------------------------
# helpers
# -------------------------------------------------
def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]

def safe_title(res):
    if res.metadata and res.metadata.get("title"):
        return res.metadata["title"].strip()
    return res.url.rstrip("/").split("/")[-1] or res.url

def safe_markdown(res):
    md = res.markdown
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    return md.raw_markdown              # Crawl4AI ≥0.6 wrapper

# -------------------------------------------------
async def crawl_site(
        root_url: str,
        depth: int,
        llms_txt: str,
        full_txt: str,
        jsonl_path: str,
):
    """Crawl → write llms.txt, llms-full.txt, train.jsonl."""

    pages, chunks = [], []          # fresh for every call

    async with AsyncWebCrawler(
        config=BrowserConfig(headless=True, verbose=False)
    ) as crawler:

        visited = set()
        current = {urldefrag(root_url)[0]}
        run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=10,
        )

        for d in range(depth):
            todo = [u for u in current if u not in visited]
            if not todo:
                break
            results = await crawler.arun_many(
                urls=todo, config=run_cfg, dispatcher=dispatcher
            )
            next_lvl = set()

            for r in results:
                visited.add(urldefrag(r.url)[0])
                if not r.success:
                    print(f"[ERROR] {r.url}: {r.error_message}")
                    continue

                md = safe_markdown(r).strip()
                pages.append({"title": safe_title(r), "url": r.url, "md": md})
                if md:
                    chunks.append(md)

                for link in r.links.get("internal", []):
                    next_lvl.add(urldefrag(link["href"])[0])

            current = next_lvl

    # -- llms.txt ----------------------------------------------------------
    with open(llms_txt, "w", encoding="utf-8") as f:
        f.write(f"# {root_url}\n")
        f.write(f"> Auto-generated on {datetime.utcnow():%Y-%m-%d}\n\n")
        for p in pages:
            f.write(f"- [{p['title']}]({p['url']}.md)\n")

    # -- llms-full.txt -----------------------------------------------------
    with open(full_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

    # -- train.jsonl  (simple summarise-each-chunk pattern) ----------------
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in pages:
            line = {
                "messages": [
                    {"role": "system", "content": "Summarise the following."},
                    {"role": "user", "content": textwrap.shorten(p["md"], 2048)},
                    {"role": "assistant", "content": p["title"]},
                ]
            }
            f.write(json.dumps(line) + "\n")

    # Return file paths so callers (Gradio etc.) can serve them
    return llms_txt, full_txt, jsonl_path
