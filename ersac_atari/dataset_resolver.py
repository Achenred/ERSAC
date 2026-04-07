# ersac_atari/dataset_resolver.py
from __future__ import annotations
import re
import difflib
from typing import Iterable, List, Optional, Tuple

try:
    import minari
except ImportError:
    minari = None


# ---- Configuration ----
# Preferred quality/version order if user doesn't specify one.
PREFERRED_ORDER = [
    "expert-v1", "expert-v0",
    "medium-replay-v2", "medium-replay-v1", "medium-replay-v0",
    "medium-v2", "medium-v1", "medium-v0",
    "random-v2", "random-v1", "random-v0",
]

# How many suggestions to show in errors.
N_SUGGESTIONS = 8

# Minimum fuzzy ratio to consider a candidate “close”.
FUZZY_THRESHOLD = 0.55


def _norm(s: str) -> str:
    """Normalize user strings for robust matching."""
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    return _norm(s).split()


def _list_remote() -> List[str]:
    if minari is None:
        raise ImportError("Please `pip install minari`.")
    try:
        return list(minari.list_remote_datasets())
    except Exception:
        # Network might be down; return empty and let local list handle it.
        return []


def _list_local() -> List[str]:
    if minari is None:
        raise ImportError("Please `pip install minari`.")
    try:
        return list(minari.list_local_datasets())
    except Exception:
        return []


def _hierarchical_like(s: str) -> bool:
    # e.g., atari/breakout/expert-v0
    return s.count("/") >= 2 and s.startswith("atari/")


def _extract_game_quality(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From a loose string like 'breakout expert v0' -> ('breakout', 'expert-v0').
    If only 'breakout' -> ('breakout', None)
    """
    toks = _tokenize(user_input)
    if not toks:
        return None, None

    # heuristic: last token(s) that look like quality variants
    quality = None
    for i in range(len(toks)):
        tail = "-".join(toks[i:])
        if re.search(r"(expert|medium|random)(-?replay)?-?v[0-9]+", tail):
            quality = re.sub(r"\s+", "-", tail)
            quality = quality.replace("--", "-")
            game = " ".join(toks[:i]) or None
            return (game, quality)

    # no explicit quality found -> all tokens are game name
    game = " ".join(toks)
    return (game, None)


def _best_quality_for_game(game: str, remote: Iterable[str], preferred_order: List[str]) -> Optional[str]:
    """
    Pick a dataset id for the given game using preferred quality order.
    """
    candidates = [d for d in remote if d.startswith(f"atari/{game}/")]
    if not candidates:
        return None
    # quick map from quality to full id
    qmap = {d.split("/")[-1]: d for d in candidates}
    for q in preferred_order:
        if q in qmap:
            return qmap[q]
    # fallback: pick any deterministic candidate
    return sorted(candidates)[0]


def _fuzzy_pick(user_str: str, universe: Iterable[str]) -> Tuple[Optional[str], List[str]]:
    """
    Fuzzy match against full dataset IDs and against "atari/<game>/*" game names.
    Returns (best_id, suggestions).
    """
    universe = list(universe)
    # Try exact first
    if user_str in universe:
        return user_str, []

    # Fuzzy on full ids
    full_sugg = difflib.get_close_matches(user_str, universe, n=N_SUGGESTIONS, cutoff=FUZZY_THRESHOLD)

    # Fuzzy on just the game portion to gather candidates
    game_names = sorted({u.split("/")[1] for u in universe if u.startswith("atari/") and u.count("/") >= 2})
    game_sugg = difflib.get_close_matches(user_str, game_names, n=N_SUGGESTIONS, cutoff=FUZZY_THRESHOLD)

    # Prioritize exact game name matches if any
    for g in game_sugg:
        # pick best quality for this game
        best = _best_quality_for_game(g, universe, PREFERRED_ORDER)
        if best:
            return best, full_sugg[:N_SUGGESTIONS]

    # fallback to closest full id
    if full_sugg:
        return full_sugg[0], full_sugg[1:N_SUGGESTIONS]

    return None, []


def resolve_minari_dataset_id(
    dataset_arg: Optional[str] = None,
    game: Optional[str] = None,
    quality: Optional[str] = None,
    preferred_order: Optional[List[str]] = None,
) -> str:
    """
    Resolve a user-provided dataset string (loose alias or full id) or (game, quality)
    into a canonical Minari dataset id. Uses remote list; falls back to local if offline.
    """
    remote = _list_remote()
    local = _list_local()
    universe = remote or local  # if offline, we still try local ids

    if not universe:
        raise RuntimeError("No Minari datasets visible (remote or local). Check network or Minari install.")

    preferred_order = preferred_order or PREFERRED_ORDER

    # Case A: user passed a full id (hierarchical)
    if dataset_arg and _hierarchical_like(dataset_arg):
        # validate against universe; allow case-insensitive
        cid = next((d for d in universe if d.lower() == dataset_arg.lower()), None)
        if cid:
            return cid
        # maybe user misspelled; fuzzy suggestions
        _, sugg = _fuzzy_pick(dataset_arg.lower(), universe)
        raise ValueError(
            f"Dataset id '{dataset_arg}' not found.\n"
            f"Suggestions: {sugg}"
        )

    # Case B: user passed a loose string (e.g., 'breakout expert')
    if dataset_arg:
        g, q = _extract_game_quality(dataset_arg)
        if g:
            g = _norm(g).replace(" ", "-")
        if q:
            q = _norm(q).replace(" ", "-")
        if g and q:
            candidate = f"atari/{g}/{q}"
            cid = next((d for d in universe if d.lower() == candidate.lower()), None)
            if cid:
                return cid
        # If no quality, pick best with preferred order
        if g and not q:
            best = _best_quality_for_game(g.replace(" ", "-"), universe, preferred_order)
            if best:
                return best

        # Fuzzy fallback
        best, sugg = _fuzzy_pick(_norm(dataset_arg).replace(" ", "-"), universe)
        if best:
            return best
        raise ValueError(
            f"Could not resolve dataset from '{dataset_arg}'. "
            f"Try a full id like 'atari/breakout/expert-v0' or one of: {sugg}"
        )

    # Case C: (game, quality) given explicitly
    if game:
        g = _norm(game).replace(" ", "-")
        if quality:
            q = _norm(quality).replace(" ", "-")
            candidate = f"atari/{g}/{q}"
            cid = next((d for d in universe if d.lower() == candidate.lower()), None)
            if cid:
                return cid
        # choose best quality if missing or invalid
        best = _best_quality_for_game(g, universe, preferred_order)
        if best:
            return best
        # fuzzy on game only
        best, sugg = _fuzzy_pick(g, universe)
        if best:
            return best
        raise ValueError(
            f"No datasets found for game '{game}'. Suggestions: {sugg}"
        )

    raise ValueError("You must provide either --dataset <str> or (--game <str> [--quality <str>]).")


def load_or_download(dataset_id: str):
    """
    Load a dataset if present locally; otherwise download it.
    Returns the Minari dataset object.
    """
    if minari is None:
        raise ImportError("Please `pip install minari`.")
    # Prefer local: avoid network if already cached.
    local = set(_list_local())
    if dataset_id in local:
        return minari.load_dataset(dataset_id)  # no remote call

    # Else programmatic download.
    return minari.load_dataset(dataset_id, download=True)
