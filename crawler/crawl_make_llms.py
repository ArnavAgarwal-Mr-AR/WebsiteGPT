import asyncio
import json
from datetime import datetime 
from urllib.parse import urldefrag
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
    MemoryAdaptiveDispatcher
)

all_pages   = []   # keep everything until we finish
all_chunks  = []   # for llms-full.txt

def safe_title(res):
    """Return page title or a fallback slug."""
    if res.metadata and res.metadata.get("title"):
        return res.metadata["title"]
    return res.url.rstrip("/").split("/")[-1] or res.url

def safe_markdown(res):
    """Return raw markdown as a string (may be None)."""
    md = res.markdown
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    return md.raw_markdown           

async def crawl_recursive_batch(start_urls, max_depth=3, max_concurrent=10):
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,      # Don't exceed 70% memory usage
        check_interval=1.0,                 # Check memory every second
        max_session_permit=max_concurrent   # Max parallel browser sessions
    )

    # Track visited URLs to prevent revisiting and infinite loops (ignoring fragments)
    visited = set()
    def normalize_url(url):
        # Remove fragment (part after #)
        return urldefrag(url)[0]
    current_urls = set([normalize_url(u) for u in start_urls])

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            print(f"\n=== Crawling Depth {depth+1} ===")
            # Only crawl URLs we haven't seen yet (ignoring fragments)
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]

            if not urls_to_crawl:
                break

            # Batch-crawl all URLs at this depth in parallel
            results = await crawler.arun_many(
                urls=urls_to_crawl,
                config=run_config,
                dispatcher=dispatcher
            )

            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)  # Mark as visited (no fragment)
                if result.success:
                    all_pages.append({
                        "title": safe_title(result),
                        "url":   result.url,
                        "md":    safe_markdown(result).strip()
                        })
                    all_chunks.append(result.markdown.strip())
                    print(f"[OK] {result.url} | Markdown: {len(result.markdown) if result.markdown else 0} chars")
                    # Collect all new internal links for the next depth
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)
                else:
                    print(f"[ERROR] {result.url}: {result.error_message}")
                    
            # Move to the next set of URLs for the next recursion depth
            current_urls = next_level_urls
        # --- write llms.txt (curated index) ---
        with open("llms.txt", "w", encoding="utf-8") as f:
            f.write(f"# {start_urls[0]}\n")
            f.write(f"> Auto-generated on {datetime.utcnow():%Y-%m-%d}\n\n")
            for p in all_pages:
                f.write(f"- [{p['title']}]({p['url']}.md)\n")

        # --- write llms-full.txt (full text) ---
        with open("llms-full.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_chunks))

data = {
    "binary": {
        "llms.txt": {"data": open("llms.txt","rb").read().decode(), "mimeType":"text/plain"},
        "llms-full.txt": {"data": open("llms-full.txt","rb").read().decode(), "mimeType":"text/plain"}
    }
}

with open('train.jsonl', 'w') as f:
    json.dump(data, f, indent=4)

if __name__ == "__main__":
    asyncio.run(crawl_recursive_batch(["https://modal.com/"], max_depth=3, max_concurrent=10))