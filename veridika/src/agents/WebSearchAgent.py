from veridika.src.agents.baseagent import BaseAgent, HistoryEntry, HistoryEntryType
from veridika.src.web_search import WebSearch


class WebSearchAgent(BaseAgent):
    def __init__(self, search_provider: str = "serper", max_history_size: int | None = None):
        """Initialize the WebSearchAgent.

        Args:
            search_provider (str): The search provider to use (default: "serper")
            max_history_size (Optional[int]): Maximum size of conversation history to maintain

        Returns:
            None
        """
        super().__init__(
            name="WebSearchAgent",
            description="An agent that performs web searches using the WebSearch module to gather information",
            max_history_size=max_history_size,
        )
        self.web_search = WebSearch(search_provider)

    def run(
        self,
        queries: list[str],
        language: str = "en",
        location: str | None = None,
        top_k: int = 10,
        ban_domains: list[str] | None = None,
        download_text: bool = True,
        download_favicon: bool = True,
    ) -> tuple[dict[str, list[dict]], float, HistoryEntry]:
        """Execute web searches for the given queries.

        Args:
            queries (List[str]): List of search queries to execute
            language (str): Language code for search results (default: "en")
            location (Optional[str]): Location code for search results (default: None)
            top_k (int): Number of results to return per query (default: 10)
            ban_domains (Optional[List[str]]): Domains to exclude from results (default: None)
            download_text (bool): Whether to download full text content (default: True)
            download_favicon (bool): Whether to download favicon content (default: True)
        Returns:
            Tuple[Dict[str, List[Dict]], float, HistoryEntry]: A tuple containing:
                - Dictionary with unified format: {"query1 | query2": [documents...]}
                  Each document has:
                  {
                      "text": "document content...",
                      "metadata": {
                          "url": "https://example.com",
                          "source": "example.com",
                          "favicon": "https://example.com/favicon.ico",
                          "title": "Document Title",
                          "type": "web_search",
                          "base_url": "https://example.com"
                      }
                  }
                - The cost of the search operations
                - History entry containing search metadata
        """
        # Prepare search parameters
        search_params = {
            "queries": queries,
            "language": language,
            "location": location,
            "top_k": top_k,
            "download_text": download_text,
            "download_favicon": download_favicon,
        }

        # Add ban_domains if provided
        if ban_domains is not None:
            search_params["ban_domains"] = ban_domains

        # Execute the web search
        raw_results, cost = self.web_search(**search_params)

        # Transform to unified format
        combined_query_key = " | ".join(queries)
        unified_results = {
            combined_query_key: [
                {
                    "text": doc.get("text", None),
                    "metadata": {
                        "url": doc["url"],
                        "source": doc["source"],
                        "favicon": doc.get("favicon", None),
                        "title": doc.get("title", ""),
                        "type": "web_search",
                        "base_url": doc.get("base_url", doc["url"]),
                        "snippet": doc.get("snippet", ""),
                    },
                }
                for doc in raw_results
            ]
        }

        # Create history entry
        history_entry = HistoryEntry(
            data={
                "searches": queries,
                "results": unified_results,
            },
            type=HistoryEntryType.web_search,
            model_metadata={
                "search_provider": self.web_search.api_name,
                "queries": queries,
                "language": language,
                "location": location,
                "top_k": top_k,
                "download_text": download_text,
                "download_favicon": download_favicon,
                "results_count": len(raw_results),
            },
        )

        return unified_results, cost, history_entry
