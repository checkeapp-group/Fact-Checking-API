from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class Metadata:
    """Immutable metadata attached to a single citation source.

    Attributes:
        source: Human‑readable publisher or site name.
        url: Canonical URL for the source.
        favicon: URL to a small icon that represents the source.
    """

    source: str
    url: str
    favicon: str


class CitationManager:
    """Store citation metadata and manipulate numbered references inside text blocks.

    The manager guarantees that every *unique* URL maps to exactly one integer
    identifier. Identifiers start at **0** (so the first item is `[0]`) and
    remain stable until :py:meth:`reorder_citations` is invoked, which
    deliberately renumbers cited ids so the first citation becomes `[1]`, the
    next `[2]`, and so on.

    Args:
        metadatas: Optional batch of metadata dictionaries to preload when the
            manager is created. Each dict must contain the keys
            ``{"source", "url", "favicon"}``.
    """

    # Pre‑compiled pattern speeds up repeated citation parsing (≈4‑10× faster)
    CITATION_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

    def __init__(self, metadatas: list[dict[str, str]] | None = None) -> None:
        self.metadatas: list[Metadata | None] = []  # sparse list; index == id
        self.url2id: dict[str, int] = {}
        self.id2metadata: dict[int, Metadata] = {}
        self._next_id: int = 0  # ids start at 0
        if metadatas:
            self.add_metadatas(metadatas)

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def add_metadatas(self, metadatas: list[dict[str, str]]) -> list[int]:
        """Insert a batch of metadata records.

        Args:
            metadatas: A list of dictionaries with the keys ``source``, ``url``,
                and ``favicon``.

        Returns:
            List of integer ids in the same order as the supplied *metadatas*.
            When a URL is already known, its existing id is returned instead of
            allocating a new one.
        """
        ids: list[int] = []
        for md in metadatas:
            url = md["url"]
            if url in self.url2id:  # duplicate — reuse id
                ids.append(self.url2id[url])
                continue
            meta_obj = Metadata(**md)  # build only if new
            cid = self._next_id
            self._next_id += 1

            # ensure sparse list is big enough
            if cid >= len(self.metadatas):
                self.metadatas.extend([None] * (cid - len(self.metadatas) + 1))
            self.metadatas[cid] = meta_obj

            self.id2metadata[cid] = meta_obj
            self.url2id[url] = cid
            ids.append(cid)
        return ids

    def get_ids(self, urls: list[str]) -> list[int]:
        """Translate URLs to their citation ids.

        Args:
            urls: List of URL strings.

        Returns:
            The ids corresponding to URLs that are already known. URLs that
            have not been added are skipped.
        """
        return [self.url2id[u] for u in urls if u in self.url2id]

    def get_metadata(self, ids: list[int]) -> list[Metadata]:
        """Retrieve :class:`Metadata` objects for a list of ids.

        Args:
            ids: Citation ids.

        Returns:
            List of :class:`Metadata` in the same order as *ids* for ids that
            are present in the manager.
        """
        return [self.id2metadata[i] for i in ids if i in self.id2metadata]

    # ------------------------------------------------------------------
    # Formatting helpers ------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def format_prompt(cid: int, text: str) -> str:
        """Generate a compact single‑line prompt entry.

        Args:
            cid: Citation id.
            text: Associated text snippet.

        Returns:
            The text formatted as ``"[<id>] <snippet>"`` with internal
            whitespace normalised to single spaces.
        """
        return f"[{cid}] {' '.join(text.split())}"

    def prepare_for_prompt(self, texts: list[str], metadatas: list[dict[str, str]]) -> str:
        """Create a deduplicated, sorted prompt block.

        Args:
            texts: Snippets that need to be referenced in the prompt.
            metadatas: Matching metadata dictionaries for each snippet.

        Returns:
            A newline‑separated string of formatted prompt lines sorted by id.
        """
        ids = self.add_metadatas(metadatas)
        seen: set[str] = set()
        formatted: list[tuple[int, str]] = []
        for text, cid in zip(texts, ids, strict=False):
            entry = self.format_prompt(cid, text)
            if entry not in seen:
                seen.add(entry)
                formatted.append((cid, entry))
        formatted.sort(key=lambda t: t[0])
        return "\n".join(e for _, e in formatted)

    def prepare_from_unified_results(self, search_results: dict[str, list[dict]]) -> str:
        """Create a deduplicated, sorted prompt block from unified search results.

        Args:
            search_results: Unified format results from WebSearch/RAG agents.
                Expected format: {"query": [{"text": "...", "metadata": {...}}, ...]}

        Returns:
            A newline‑separated string of formatted prompt lines sorted by id.
        """
        texts = []
        metadatas = []
        for query_key, documents in search_results.items():
            for doc in documents:
                texts.append(doc["text"])
                metadatas.append(
                    {
                        "source": doc["metadata"]["source"],
                        "url": doc["metadata"]["url"],
                        "favicon": doc["metadata"]["favicon"],
                    }
                )

        return self.prepare_for_prompt(texts, metadatas)

    # ------------------------------------------------------------------
    # Citation parsing & HTML helpers ----------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def retrieve_ids_from_text(cls, text: str) -> list[int]:
        """Extract all citation ids that appear in *text*.

        Args:
            text: Arbitrary string that may contain tokens like ``[1, 2]``.

        Returns:
            Sorted list of distinct ids found in the string (duplicates and
            badly‑formatted entries are ignored).
        """
        ids: set[int] = set()
        for citation in cls.CITATION_RE.findall(text):
            for num in citation.split(","):
                num = num.strip()
                if num.isdigit():
                    ids.add(int(num.lstrip("0") or "0"))  # handle leading zeros
        return sorted(ids)

    def get_html_citations(self, ids: list[int]) -> list[str]:
        """Render clickable HTML badges for a list of citation ids.

        Args:
            ids: Citation identifiers to render.

        Returns:
            A list of HTML strings; ids that do not exist in the manager are
            silently skipped.
        """
        return [
            (
                f'<a href="{meta.url}" '
                f'style="display:inline-flex;align-items:center;background-color:#4CAF50;'
                f"color:white;padding:8px 12px;border-radius:12px;text-decoration:none;"
                f'font-weight:bold;margin:5px;font-size:14px;">'
                f"[{cid}] {meta.source} "
                f'<img src="{meta.favicon}" alt="{meta.source}" '
                f'style="width:16px;height:16px;margin-left:8px;"></a>'
            )
            for cid, meta in ((i, self.id2metadata[i]) for i in ids if i in self.id2metadata)
        ]

    # ------------------------------------------------------------------
    # Renumbering -------------------------------------------------------
    # ------------------------------------------------------------------
    def reorder_citations(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Renumber citations by order of first appearance and rewrite *texts*.

        Args:
            texts: A list of documents that may contain citation markers like
                ``[3, 5]`` or ``[7]``. The list is **not** modified in‑place;
                a rewritten copy is returned.

        Returns:
            * **new_texts** – The input *texts* with all citation markers
              rewritten to consecutive ids starting at ``1``.
            * **cited_ids** – Sorted list of ids that were actually cited after
              renumbering (ids present in the doc set).
        """
        original = self.id2metadata.copy()

        # 1. Determine first‑appearance order
        ordered: list[int] = []
        seen: set[int] = set()
        for t in texts:
            for citation in self.CITATION_RE.findall(t):
                for num in citation.split(","):
                    n = num.strip()
                    if n.isdigit():
                        cid = int(n.lstrip("0") or "0")
                        if cid in original and cid not in seen:
                            seen.add(cid)
                            ordered.append(cid)

        new_id_for_old: dict[int, int] = {old: new for new, old in enumerate(ordered, 1)}

        # Include never‑cited entries so mapping remains exhaustive
        next_new = len(new_id_for_old) + 1
        for old in sorted(original):
            if old not in new_id_for_old:
                new_id_for_old[old] = next_new
                next_new += 1

        # Helper replaces a single citation token
        def _replace(match: re.Match[str]) -> str:
            old_ids = [
                int(n.lstrip("0") or "0")
                for n in match.group(1).split(",")
                if n.strip().isdigit() and int(n.lstrip("0") or "0") in original
            ]
            if not old_ids:
                return ""  # remove completely when no valid ids remain
            # Map → dedupe → sort ascending
            new_ids = sorted({new_id_for_old[o] for o in old_ids})
            return f"[{','.join(map(str, new_ids))}]"

        new_texts: list[str] = []
        for txt in texts:
            rewritten = self.CITATION_RE.sub(_replace, txt)
            # Compress runs of >1 spaces that may appear after removal of tokens
            rewritten = re.sub(r" {2,}", " ", rewritten)
            # Trim trailing spaces at line ends
            # rewritten = re.sub(r" +\n", "\n", rewritten).strip(" ")
            rewritten = rewritten.strip(" ")
            new_texts.append(rewritten)

        cited_ids = sorted({new_id_for_old[oid] for oid in seen})

        # 3. Rebuild internal structures to reflect the new id space
        self.id2metadata = {new_id_for_old[old]: meta for old, meta in original.items()}
        self.url2id = {meta.url: nid for nid, meta in self.id2metadata.items()}
        self._next_id = max(self.id2metadata, default=-1) + 1

        # Rebuild sparse list so index == id for quick lookup if ever needed
        self.metadatas = [None] * self._next_id
        for cid, meta in self.id2metadata.items():
            if cid >= len(self.metadatas):
                self.metadatas.extend([None] * (cid - len(self.metadatas) + 1))
            self.metadatas[cid] = meta

        return new_texts, cited_ids
