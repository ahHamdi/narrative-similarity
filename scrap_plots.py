import requests
import pandas as pd
import time
from tqdm import tqdm
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "WikiFilmDataset/1.2 (research; contact=example@email.com)"}

LANGS = {
    "fr": ["Synopsis"],
    "es": ["Sinopsis"],
    "pt": ["Sinopse"],
    "de": ["Handlung"],
    "it": ["Trama"],
}


def safe_get_json(url, params):
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        return resp.json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def get_movies_by_year(start=1990, end=2020, limit=10000):
    api = "https://en.wikipedia.org/w/api.php"
    films = []

    for year in range(start, end + 1):
        cont = None
        category = f"{year} films"

        while True:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmnamespace": 0,
                "cmlimit": 500,
            }
            if cont:
                params["cmcontinue"] = cont

            r = safe_get_json(api, params)
            if not r:
                break

            films.extend([m["title"] for m in r["query"]["categorymembers"]])

            cont = r.get("continue", {}).get("cmcontinue")
            if not cont or len(films) >= limit:
                break

        if len(films) >= limit:
            break

        time.sleep(0.2)

    return films[:limit]


def get_section_index(title, lang, section_names):
    api = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "sections",
        "redirects": True,
    }

    r = safe_get_json(api, params)
    if not r:
        return None

    for s in r.get("parse", {}).get("sections", []):
        if s["line"].strip().lower() in [n.lower() for n in section_names]:
            return s["index"]

    return None


def get_section_text(title, lang, section_index):
    api = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "section": section_index,
        "prop": "text",
        "redirects": True,
    }

    r = safe_get_json(api, params)
    if not r:
        return None

    html = r.get("parse", {}).get("text", {}).get("*", "")
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    text = "\n".join(p.get_text() for p in soup.find_all("p"))

    return text.strip()


def get_langlinks(title):
    api = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "langlinks",
        "titles": title,
        "lllimit": 500,
    }

    r = safe_get_json(api, params)
    if not r:
        return {}

    page = next(iter(r.get("query", {}).get("pages", {}).values()), {})
    return {l["lang"]: l["*"] for l in page.get("langlinks", [])}


def build_dataset(max_films=10000):
    rows = []
    films = get_movies_by_year(1990, 2020, max_films)

    for film in tqdm(films, desc="Films"):
        idx_en = get_section_index(film, "en", ["Plot", "Plot summary"])
        if not idx_en:
            continue

        plot_en = get_section_text(film, "en", idx_en)
        if not plot_en or len(plot_en) < 300:
            continue

        links = get_langlinks(film)

        for lang, section_names in LANGS.items():
            if lang not in links:
                continue

            idx = get_section_index(links[lang], lang, section_names)
            if not idx:
                continue

            plot_other = get_section_text(links[lang], lang, idx)
            if not plot_other or len(plot_other) < 300:
                continue

            rows.append({"film": film, "plot_en": plot_en, "plot_other": plot_other})

        time.sleep(0.25)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset(10000)
    df.to_csv("films_plots_aligned.csv", index=False)
    print(f"Dataset généré : {len(df)} lignes")
