import os
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import feedparser
from openai import OpenAI
from dateutil import parser
import re
import unicodedata
from perplexity import Perplexity
import numpy as np
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config import TRUSTED_DOMAINS, STOP_TITLE_TERMS

load_dotenv()
# Basic logging configuration for better visibility in CLI runs
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


def _get_domain(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # Normalize common subdomain prefixes we don't want to treat as separate orgs
    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def _is_trusted_newspaper(url: str) -> bool:
    if not url:
        return False
    domain = _get_domain(url)
    if "wikipedia.org" in domain:
        return False
    for trusted in TRUSTED_DOMAINS:
        if "/" in trusted:
            # Path-based trusted entry (e.g., bbc.com/mundo)
            if trusted in url:
                return True
        else:
            if domain.endswith(trusted):
                return True
    return False


def _generate_query_variants(query: str) -> List[str]:
    # Minimal, deterministic variants to improve recall without external API
    def strip_accents(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    base = query.strip()
    variants: List[str] = [base]
    sa = strip_accents(base)
    if sa != base:
        variants.append(sa)

    # Heuristic synonyms/wording
    lowered = sa.lower()
    if "flotilla" in lowered:
        variants.append(lowered.replace("flotilla", "flotilla de la libertad"))
    variants.append("israel intercepta flotilla gaza")
    variants.append("israel intercepts gaza flotilla")

    # Deduplicate preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _build_compact_prompt(query: str, target_results: int) -> str:
    cushion = max(target_results + 5, int(target_results * 1.3))
    avoid = ", ".join([f"'{t}'" for t in STOP_TITLE_TERMS])
    
    # 1. PREFIJO: Introduce el tema, pero no incluye la consulta
    base_prefix = f"Titulares recientes (<=30 días) en español sobre: '" 
    
    # 2. SUFIJO: Contiene todos los filtros y el límite.
    base_suffix = (
        f"'. Devuelve título y URL. Evita títulos con {avoid}. Máx {cushion}."
    )
    
    max_len = 256
    
    # **AQUÍ SE CALCULA CUÁNTO ESPACIO TIENE LA CONSULTA**
    remaining = max_len - (len(base_prefix) + len(base_suffix))
    
    trimmed_query = query.strip()
    
    if remaining < len(trimmed_query):
        trimmed_query = trimmed_query[: max(0, remaining - 3)].rstrip() + ("..." if remaining > 3 else "")
        
    # 3. CONCATENACIÓN: Coloca el query recortado entre el prefijo y el sufijo.
    return f"{base_prefix}{trimmed_query}{base_suffix}"

def _extract_best_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Prefer OpenGraph title
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og.get("content").strip()
    # Then the page <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # Fall back to first h1
    h1 = soup.find("h1")
    if h1 and h1.get_text():
        return h1.get_text(strip=True)
    return ""


def _fetch_full_title_from_url(url: str, timeout_seconds: int = 5) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Newssearcher/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout_seconds)
        if resp.ok and resp.text:
            return _extract_best_title(resp.text)
    except Exception as e:
        logging.warning(f"Failed to fetch title from {url}: {e}")
    return ""


def _fetch_page(url: str, timeout_seconds: int = 5) -> Tuple[str, Dict[str, str]]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Newssearcher/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout_seconds)
        if resp.ok:
            return resp.text or "", dict(resp.headers)
    except Exception as e:
        logging.warning(f"Failed to fetch page {url}: {e}")
    return "", {}

def _extract_canonical_url(html: str) -> Optional[str]:
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        link = soup.find("link", rel=lambda v: v and "canonical" in v)
        if link and link.get("href"):
            return link.get("href").strip()
    except Exception:
        return None
    return None

def _normalize_amp_in_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme, netloc, path, params, query, fragment = (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
        for sub in ("amp.", "m."):
            if netloc.startswith(sub):
                netloc = netloc[len(sub):]
        if path.endswith("/amp"):
            path = path[:-4]
        if path.endswith(".amp"):
            path = path[:-4]
        if path.endswith(".amp.html"):
            path = path.replace(".amp.html", ".html")
        if query:
            for flag in ("outputType=amp", "amp=true", "amp=1"):
                query = query.replace(flag, "")
            query = query.strip("&")
        normalized = f"{scheme}://{netloc}{path}"
        if params:
            normalized += f";{params}"
        if query:
            normalized += f"?{query}"
        return normalized
    except Exception:
        return url

def _normalize_url(url: str, html: str) -> str:
    canonical = _extract_canonical_url(html)
    base = canonical if canonical else url
    return _normalize_amp_in_url(base)

def _is_live_title(title: str) -> bool:
    if not title:
        return False
    lowered = title.lower()
    live_terms = [
        "en directo",
        "directo:",
        "en vivo",
        "minuto a minuto",
        "última hora",
        "ultima hora",
        "live:",
        "directo |",
        "en directo |",
    ]
    return any(term in lowered for term in live_terms)


def _extract_publish_datetime(html: str) -> Optional[datetime]:
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []
    # Common meta tags
    for selector, attr in [
        ("meta[property='article:published_time']", "content"),
        ("meta[property='og:article:published_time']", "content"),
        ("meta[name='pubdate']", "content"),
        ("meta[name='date']", "content"),
        ("time[datetime]", "datetime"),
    ]:
        el = soup.select_one(selector)
        if el and el.get(attr):
            candidates.append(el.get(attr))

    # Try simple JSON-LD extraction
    if not candidates:
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                text = script.string or script.get_text(" ")
                if not text:
                    continue
                # naive find to avoid heavy json parsing
                m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', text)
                if m:
                    candidates.append(m.group(1))
                    break
            except Exception:
                continue

    for raw in candidates:
        try:
            dt = parser.parse(raw)
            # Normalize to timezone-aware UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            continue
    return None


def _extract_publish_datetime_from_headers(headers: Dict[str, str]) -> Optional[datetime]:
    try:
        last_modified = headers.get("Last-Modified") or headers.get("last-modified")
        if last_modified:
            dt = parser.parse(last_modified)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
    except Exception:
        return None
    return None


def _extract_publish_datetime_from_url(url: str) -> Optional[datetime]:
    # Try patterns like /YYYY/MM/DD/ or /YYYYMMDD/
    try:
        m = re.search(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})/", url)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(year, month, day, tzinfo=timezone.utc)
        m2 = re.search(r"/(20\d{2})(\d{2})(\d{2})/", url)
        if m2:
            year, month, day = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            return datetime(year, month, day, tzinfo=timezone.utc)
    except Exception:
        return None
    return None

def _is_live_content(title: str, url: str) -> bool:
    # 1. Comprueba el título (usando tu función actual)
    if _is_live_title(title): 
        return True
    
    # 2. Comprueba la URL
    lowered_url = url.lower()
    url_live_terms = [
        "/directo",
        "/en-vivo",
        "/minuto-a-minuto",
        "ultima-hora", # Sin el guión bajo si la URL está limpia
        "liveblog",
        "/live",
        "en-directo",
        "minutoaminuto"
    ]
    return any(term in lowered_url for term in url_live_terms)
    
def search_headlines(query, fetch_timeout_seconds: int = 12, target_results: int = 20) -> List[Tuple[str, str, Optional[datetime]]]:
    """
    Searches for headlines using the Perplexity API and returns (title, url) tuples
    filtered to trusted newspaper domains. Tries to expand titles by fetching pages.
    """
    try:
        client = Perplexity(api_key=PERPLEXITY_API_KEY)
        results: List[Tuple[str, str, Optional[datetime]]] = []
        seen_urls: Set[str] = set()
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        # Build a Spanish prompt to steer Perplexity to return recent Spanish news with exact URLs
        prompt = _build_compact_prompt(query, target_results)
        raw_variants = [prompt] + _generate_query_variants(query)
        # Ensure all queries are <= 256 chars (Perplexity limit)
        query_variants: List[str] = []
        for v in raw_variants:
            if len(v) > 256:
                # Fallback: keep only the first 256 chars
                v = v[:256]
            query_variants.append(v)

        for q in query_variants:
            logging.info(f"Perplexity query: {q}")
            response = client.search.create(query=q)
            for item in getattr(response, "results", []) or []:
                title = getattr(item, "title", "").strip()
                url = (
                    getattr(item, "url", None)
                    or getattr(item, "link", None)
                    or getattr(getattr(item, "source", object()), "url", None)
                )

                if not url or not _is_trusted_newspaper(url):
                    continue

                if url in seen_urls:
                    continue

                # Fetch and parse HTML in parallel batches later for performance
                html, headers = _fetch_page(url, timeout_seconds=fetch_timeout_seconds)
                full_title = _extract_best_title(html) or title
                # Skip likely live pages as an extra safeguard
                if _is_live_content(full_title, url): # Ahora se comprueba el título Y la URL
                    logging.info(f"Skipping live content: {full_title[:50]}...")
                    continue

                # Normalize URL (canonical + amp stripping) for dedupe
                normalized_url = _normalize_url(url, html)
                if normalized_url in seen_urls:
                    continue
                published_dt = (
                    _extract_publish_datetime(html)
                    or _extract_publish_datetime_from_headers(headers)
                    or _extract_publish_datetime_from_url(url)
                )

                # Include if date is unknown; if known, require within last 30 days
                if published_dt is not None and published_dt < cutoff:
                    continue

                seen_urls.add(normalized_url)
                if full_title:
                    results.append((full_title, normalized_url, published_dt))

            # No inner break; allow this query to exhaust its results before moving on
            if len(results) >= target_results:
                break

        # Parallel enrichment step: refetch/parse in threads for any entries lacking data
        def enrich(entry: Tuple[str, str, Optional[datetime]]) -> Tuple[str, str, Optional[datetime]]:
            t, u, dt_val = entry
            if dt_val is not None and t:
                return entry
            html, headers = _fetch_page(u, timeout_seconds=fetch_timeout_seconds)
            new_title = _extract_best_title(html) or t
            new_dt = dt_val or (
                _extract_publish_datetime(html)
                or _extract_publish_datetime_from_headers(headers)
                or _extract_publish_datetime_from_url(u)
            )
            return (new_title, u, new_dt)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(enrich, e) for e in results]
            enriched: List[Tuple[str, str, Optional[datetime]]] = []
            for f in as_completed(futures):
                try:
                    enriched.append(f.result())
                except Exception as e:
                    logging.error(f"Enrichment failed: {e}")
        return enriched
    except Exception as e:
        print(f"An error occurred with the Perplexity API: {e}")
        return []

def get_embeddings(texts, model="text-embedding-ada-002"):
    """
    Gets embeddings for a list of texts using OpenAI API.
    """
    try:
        client = OpenAI()
        response = client.embeddings.create(input=texts, model=model)
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return []

def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    # Check for API keys in environment variables
    if "PERPLEXITY_API_KEY" not in os.environ:
        print("Please set the PERPLEXITY_API_KEY environment variable.")
    elif "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        user_query = input("Enter a topic to search for headlines: ")
        title_url_pairs = search_headlines(user_query, fetch_timeout_seconds=5, target_results=20)

        if title_url_pairs:
            print("\nFound headlines (Unordered):")
            for i, (headline, url, published_dt) in enumerate(title_url_pairs):
                print("-" * 80)
                print(f"{i+1}. {headline}")
                print(f"   Source: {_get_domain(url)}")
                if published_dt:
                    print(f"   Date: {published_dt.isoformat()}")
                print(f"   {url}")

            # Get embeddings for the query and headlines
            headlines_only = [t for (t, _, __) in title_url_pairs]
            all_texts = [user_query] + headlines_only
            embeddings = get_embeddings(all_texts)

            if embeddings:
                query_embedding = embeddings[0]
                headline_embeddings = embeddings[1:]

                # Calculate similarities
                similarities = [cosine_similarity(query_embedding, h_emb) for h_emb in headline_embeddings]

                # Sort headlines by similarity
                sorted_pairs = [pair for _, pair in sorted(zip(similarities, title_url_pairs), reverse=True)]

                print("\nHeadlines sorted by similarity:")
                for i, (headline, url, published_dt) in enumerate(sorted_pairs):
                    print("-" * 80)
                    print(f"{i+1}. {headline}")
                    print(f"   Source: {_get_domain(url)}")
                    if published_dt:
                        print(f"   Date: {published_dt.isoformat()}")
                    print(f"   {url}")