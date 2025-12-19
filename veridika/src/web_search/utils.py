# Add necessary imports for all content extraction libraries
import asyncio

# Content extraction methods enum
from enum import Enum
import logging
import multiprocessing
import re
import threading
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

# Goose3
from goose3 import Goose

# Newspaper3k
from newspaper import Article

# Readability
from readability import Document
import requests
from trafilatura import extract
from trafilatura.settings import use_config


def is_valid_url(url):
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def clean_url(full_url):
    """
    Clean a URL by removing any query parameters.

    Args:
        full_url (str): The full URL.

    Returns:
        str: The cleaned URL.
    """
    # Remove special characters
    full_url = full_url.encode("ascii", "ignore").decode("ascii")
    # Remove query parameters
    parsed_url = urlparse(full_url)
    # Build the base URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    return base_url


def get_domain_name(url: str) -> str:
    """
    Get the domain name from a URL.

    Args:
        url (str): The URL.

    Returns:
        str: The domain name.
    """

    parsed_uri = urlparse(url)
    return f"{parsed_uri.netloc}"


def clean_text(text):
    """
    Clean the text by removing extra whitespace and newlines.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = " ".join(text.strip().split()).strip()
    return text


def run_async_in_thread(coroutine):
    result = []
    exception = []

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.append(loop.run_until_complete(coroutine))
        except Exception as e:
            exception.append(e)
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if exception:
        raise exception[0]
    return result[0]


class ExtractMethod(Enum):
    TRAFILATURA = "trafilatura"
    NEWSPAPER = "newspaper"
    READABILITY = "readability"
    GOOSE = "goose"
    BEAUTIFULSOUP = "beautifulsoup"  # Original method as fallback


def extract_content(html_content, url, method=ExtractMethod.NEWSPAPER):
    """
    Extract content from HTML using the specified method.

    Args:
        html_content (str): HTML content of the page
        url (str): URL of the page (needed for some extractors)
        method (ExtractMethod): Method to use for extraction

    Returns:
        Tuple[str, str]: Title and extracted text
    """
    title = None
    text = None
    fallback_title = "No Title Found"  # Default fallback title

    try:
        if method == ExtractMethod.TRAFILATURA:
            # Use trafilatura for content extraction
            traf_config = use_config()
            text = extract(
                html_content,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                output_format="txt",
                config=traf_config,
            )
            # Trafilatura doesn't reliably extract titles, so use BeautifulSoup for title as fallback
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.title.string.strip() if soup.title else fallback_title

        elif method == ExtractMethod.NEWSPAPER:
            # Use newspaper3k
            article = Article(url)
            article.set_html(html_content)
            article.parse()
            title = article.title
            text = article.text
            if not title:  # Fallback title if newspaper fails
                soup = BeautifulSoup(html_content, "html.parser")
                title = soup.title.string.strip() if soup.title else fallback_title

        elif method == ExtractMethod.READABILITY:
            # Use readability-lxml
            doc = Document(html_content)
            title = doc.title()
            content_html = doc.summary()
            # Strip HTML tags for plain text
            content_soup = BeautifulSoup(content_html, "html.parser")
            text = content_soup.get_text(separator="\\n").strip()
            if not title:  # Fallback title if readability fails
                soup = BeautifulSoup(html_content, "html.parser")
                title = soup.title.string.strip() if soup.title else fallback_title

        elif method == ExtractMethod.GOOSE:
            # Use goose3
            g = Goose()
            article = g.extract(raw_html=html_content)
            title = article.title
            text = article.cleaned_text
            if not title:  # Fallback title if goose fails
                soup = BeautifulSoup(html_content, "html.parser")
                title = soup.title.string.strip() if soup.title else fallback_title

        elif method == ExtractMethod.BEAUTIFULSOUP:
            # Use BeautifulSoup only when explicitly requested
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.title.string.strip() if soup.title else fallback_title
            text_elements = [clean_text(p.text.strip()) for p in soup.select("p")]
            text = "\\n".join(filter(None, text_elements))

    except Exception as e:
        logging.info(f"Error with {method.value} extraction: {e}. Falling back to BeautifulSoup.")
        # Fall back to BeautifulSoup if the chosen method fails
        soup = BeautifulSoup(html_content, "html.parser")
        title = soup.title.string.strip() if soup.title else fallback_title
        text_elements = [clean_text(p.text.strip()) for p in soup.select("p")]
        text = "\\n".join(filter(None, text_elements))

    # Ensure title is never None before returning
    if title is None:
        title = fallback_title

    return title, text


async def download_text_and_title_async(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    extract_method=ExtractMethod.TRAFILATURA,
) -> tuple[str, str, str]:
    """
    Download title and text from a URL using the specified extraction method,
    with a fallback to requests if the initial download fails.
    """
    initial_url = url.encode("ascii", "ignore").decode("ascii")
    title, extracted_text, final_url = None, None, None

    async with semaphore:
        try:
            # Primary attempt with aiohttp
            # Session is passed in, headers are configured there
            async with session.get(initial_url, allow_redirects=True, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    final_url = str(response.url)
                    title, extracted_text = extract_content(content, final_url, extract_method)
                else:
                    logging.info(
                        f"aiohttp request failed for {initial_url} with status {response.status}. Trying requests fallback."
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message="Non-200 status",
                    )

        except (TimeoutError, aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            logging.info(
                f"aiohttp request failed for {initial_url}: {e}. Trying requests fallback."
            )
            # Fallback attempt with requests (synchronous, needs run_in_executor)
            try:
                loop = asyncio.get_running_loop()
                # Get headers from the aiohttp session
                headers = session.headers

                # Run requests.get in an executor
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        initial_url, headers=headers, timeout=10, allow_redirects=True
                    ),
                )

                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

                content = response.text
                final_url = response.url  # Final URL after redirects
                title, extracted_text = extract_content(content, final_url, extract_method)

                logging.info(f"Requests fallback succeeded for {initial_url}")

            except requests.exceptions.RequestException as req_e:
                logging.info(f"Requests fallback also failed for {initial_url}: {req_e}")
                title, extracted_text, final_url = None, None, None
            except Exception as fallback_e:
                logging.info(
                    f"Error during requests fallback processing for {initial_url}: {fallback_e}"
                )
                title, extracted_text, final_url = None, None, None

        except Exception as e:
            logging.info(
                f"Unexpected error during aiohttp processing for {initial_url}: {e}. No fallback attempted."
            )
            title, extracted_text, final_url = None, None, None

    return title, extracted_text, final_url


async def download_text_and_title_parallel_async(
    urls: list[str], extract_method=ExtractMethod.NEWSPAPER
) -> list[tuple[str, str, str]]:
    """
    Download title and text from a list of URLs in parallel using the specified extraction method.

    Args:
        urls (List[str]): The list of URLs to download.
        extract_method (ExtractMethod): Method to use for content extraction.

    Returns:
        List[Tuple[str, str, str]]: The list of title, text, and final URL tuples.
    """
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    # Define headers for aiohttp session
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            download_text_and_title_async(url, session, semaphore, extract_method) for url in urls
        ]
        results = await asyncio.gather(*tasks)

    return results


def download_text_and_title_parallel(
    urls: list[str], extract_method=ExtractMethod.NEWSPAPER
) -> list[tuple[str, str, str]]:
    """
    Wrapper function to run the async function in both synchronous and asynchronous contexts.
    """
    return run_async_in_thread(download_text_and_title_parallel_async(urls, extract_method))


async def download_text_title_favicon_async(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    extract_method=ExtractMethod.NEWSPAPER,
) -> tuple[str, str, str, str]:
    """
    Download title, text, final URL, and favicon from a URL asynchronously,
    with a fallback to requests if the initial download fails.
    """
    initial_url = url.encode("ascii", "ignore").decode("ascii")
    title, extracted_text, final_url, favicon_url = None, None, None, None
    default_favicon = "https://upload.wikimedia.org/wikipedia/commons/0/01/Website_icon.svg"

    async with semaphore:
        try:
            # Primary attempt with aiohttp
            # Session is passed in, headers are configured there
            async with session.get(initial_url, allow_redirects=True, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    final_url = str(response.url)

                    # Extract content using specified method
                    title, extracted_text = extract_content(content, final_url, extract_method)

                    # Extract favicon using BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")
                    favicon_url = default_favicon  # Start with default
                    icon_links = soup.find_all(
                        "link",
                        rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                    )
                    meta_icons = soup.find_all(
                        "meta", attrs={"content": re.compile(r".ico$", re.I)}
                    )
                    icons = icon_links + meta_icons

                    for icon in icons:
                        found_url = icon.get("href") or icon.get("content")
                        if found_url:
                            # Ensure the URL is absolute
                            parsed_found_url = urlparse(found_url)
                            if not parsed_found_url.scheme:
                                favicon_url = urljoin(
                                    final_url, found_url
                                )  # Make absolute relative to final URL
                            else:
                                favicon_url = found_url  # It's already absolute
                            break  # Use the first one found

                else:
                    logging.info(
                        f"aiohttp request failed for {initial_url} with status {response.status}. Trying requests fallback."
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message="Non-200 status",
                    )

        except (TimeoutError, aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            logging.info(
                f"aiohttp request failed for {initial_url}: {e}. Trying requests fallback."
            )
            # Fallback attempt with requests
            try:
                loop = asyncio.get_running_loop()
                headers = session.headers

                # Run requests.get in an executor
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        initial_url, headers=headers, timeout=10, allow_redirects=True
                    ),
                )

                response.raise_for_status()  # Check for HTTP errors

                content = response.text
                final_url = response.url

                # Extract content using specified method
                title, extracted_text = extract_content(content, final_url, extract_method)

                # Extract favicon using BeautifulSoup from requests content
                soup = BeautifulSoup(content, "html.parser")
                favicon_url = default_favicon  # Start with default
                icon_links = soup.find_all(
                    "link",
                    rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                )
                meta_icons = soup.find_all("meta", attrs={"content": re.compile(r".ico$", re.I)})
                icons = icon_links + meta_icons

                for icon in icons:
                    found_url = icon.get("href") or icon.get("content")
                    if found_url:
                        parsed_found_url = urlparse(found_url)
                        if not parsed_found_url.scheme:
                            favicon_url = urljoin(final_url, found_url)
                        else:
                            favicon_url = found_url
                        break

                logging.info(f"Requests fallback succeeded for {initial_url}")

            except requests.exceptions.RequestException as req_e:
                logging.info(f"Requests fallback also failed for {initial_url}: {req_e}")
                title, extracted_text, final_url, favicon_url = None, None, None, None
            except Exception as fallback_e:
                logging.info(
                    f"Error during requests fallback processing for {initial_url}: {fallback_e}"
                )
                title, extracted_text, final_url, favicon_url = None, None, None, None

        except Exception as e:
            logging.info(
                f"Unexpected error during aiohttp processing for {initial_url}: {e}. No fallback attempted."
            )
            title, extracted_text, final_url, favicon_url = None, None, None, None

    # Ensure a default favicon if extraction failed but text/title might exist from fallback
    if title and not favicon_url:
        favicon_url = default_favicon

    return title, extracted_text, final_url, favicon_url


async def download_text_favicon_parallel_async(
    urls: list[str], extract_method=ExtractMethod.NEWSPAPER
) -> list[tuple[str, str, str, str]]:
    """
    Download title, text, final URL, and favicon from a list of URLs in parallel using the specified extraction method.
    """
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    # Define headers for aiohttp session
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = {
            url: asyncio.create_task(
                download_text_title_favicon_async(url, session, semaphore, extract_method)
            )
            for url in urls
        }

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        # Reorder results based on input URL order
        results = []
        for url in urls:
            result = await tasks[url]
            results.append(result)

    return results


def download_text_favicon_parallel(
    urls: list[str], extract_method=ExtractMethod.NEWSPAPER
) -> list[tuple[str, str, str, str]]:
    """
    Wrapper function to run the async function in both synchronous and asynchronous contexts.
    """
    return run_async_in_thread(download_text_favicon_parallel_async(urls, extract_method))


async def download_favicon_async(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """
    Download only the favicon from a URL asynchronously,
    with a fallback to requests if the initial download fails.

    Returns:
        Tuple[str, str]: final_url and favicon_url
    """
    initial_url = url.encode("ascii", "ignore").decode("ascii")
    final_url, favicon_url = None, None
    default_favicon = "https://upload.wikimedia.org/wikipedia/commons/0/01/Website_icon.svg"

    async with semaphore:
        try:
            # Primary attempt with aiohttp
            async with session.get(initial_url, allow_redirects=True, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    final_url = str(response.url)

                    # Extract favicon using BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")
                    favicon_url = default_favicon  # Start with default
                    icon_links = soup.find_all(
                        "link",
                        rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                    )
                    meta_icons = soup.find_all(
                        "meta", attrs={"content": re.compile(r".ico$", re.I)}
                    )
                    icons = icon_links + meta_icons

                    for icon in icons:
                        found_url = icon.get("href") or icon.get("content")
                        if found_url:
                            # Ensure the URL is absolute
                            parsed_found_url = urlparse(found_url)
                            if not parsed_found_url.scheme:
                                favicon_url = urljoin(
                                    final_url, found_url
                                )  # Make absolute relative to final URL
                            else:
                                favicon_url = found_url  # It's already absolute
                            break  # Use the first one found

                else:
                    logging.info(
                        f"aiohttp request failed for {initial_url} with status {response.status}. Trying requests fallback."
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message="Non-200 status",
                    )

        except (TimeoutError, aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            logging.info(
                f"aiohttp request failed for {initial_url}: {e}. Trying requests fallback."
            )
            # Fallback attempt with requests
            try:
                loop = asyncio.get_running_loop()
                headers = session.headers

                # Run requests.get in an executor
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        initial_url, headers=headers, timeout=10, allow_redirects=True
                    ),
                )

                response.raise_for_status()  # Check for HTTP errors

                content = response.text
                final_url = response.url

                # Extract favicon using BeautifulSoup from requests content
                soup = BeautifulSoup(content, "html.parser")
                favicon_url = default_favicon  # Start with default
                icon_links = soup.find_all(
                    "link",
                    rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                )
                meta_icons = soup.find_all("meta", attrs={"content": re.compile(r".ico$", re.I)})
                icons = icon_links + meta_icons

                for icon in icons:
                    found_url = icon.get("href") or icon.get("content")
                    if found_url:
                        parsed_found_url = urlparse(found_url)
                        if not parsed_found_url.scheme:
                            favicon_url = urljoin(final_url, found_url)
                        else:
                            favicon_url = found_url
                        break

                logging.info(f"Requests fallback succeeded for {initial_url}")

            except requests.exceptions.RequestException as req_e:
                logging.info(f"Requests fallback also failed for {initial_url}: {req_e}")
                final_url, favicon_url = None, None
            except Exception as fallback_e:
                logging.info(
                    f"Error during requests fallback processing for {initial_url}: {fallback_e}"
                )
                final_url, favicon_url = None, None

        except Exception as e:
            logging.info(
                f"Unexpected error during aiohttp processing for {initial_url}: {e}. No fallback attempted."
            )
            final_url, favicon_url = None, None

    # Ensure a default favicon if extraction failed
    if final_url and not favicon_url:
        favicon_url = default_favicon

    return final_url, favicon_url


async def download_favicon_parallel_async(urls: list[str]) -> list[tuple[str, str]]:
    """
    Download favicons from a list of URLs in parallel.

    Args:
        urls (List[str]): The list of URLs to download favicons from.

    Returns:
        List[Tuple[str, str]]: The list of (final_url, favicon_url) tuples.
    """
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    # Define headers for aiohttp session
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = {
            url: asyncio.create_task(download_favicon_async(url, session, semaphore))
            for url in urls
        }

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        # Reorder results based on input URL order
        results = []
        for url in urls:
            result = await tasks[url]
            results.append(result)

    return results


def download_favicon_parallel(urls: list[str]) -> list[tuple[str, str]]:
    """
    Download favicons from a list of URLs in parallel.

    Args:
        urls (List[str]): The list of URLs to download favicons from.

    Returns:
        List[Tuple[str, str]]: The list of (final_url, favicon_url) tuples.
    """
    return run_async_in_thread(download_favicon_parallel_async(urls))


async def download_content_async(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    download_text: bool = True,
    download_title: bool = True,
    download_favicon: bool = True,
    extract_method: ExtractMethod = ExtractMethod.NEWSPAPER,
) -> tuple[str, str, str, str]:
    """
    Download selectively from a URL asynchronously based on specified parameters,
    with a fallback to requests if the initial download fails.

    Args:
        url (str): The URL to download from
        session (aiohttp.ClientSession): The aiohttp session
        semaphore (asyncio.Semaphore): Semaphore for connection limiting
        download_text (bool): Whether to extract text content
        download_title (bool): Whether to extract title
        download_favicon (bool): Whether to extract favicon
        extract_method (ExtractMethod): Method to use for content extraction

    Returns:
        Tuple[str, str, str, str]: title, extracted_text, final_url, favicon_url
                                  (None for disabled options)
    """

    initial_url = url.encode("ascii", "ignore").decode("ascii")
    title, extracted_text, final_url, favicon_url = None, None, None, None
    default_favicon = "https://upload.wikimedia.org/wikipedia/commons/0/01/Website_icon.svg"
    fallback_title = "No Title Found"

    async with semaphore:
        try:
            # Primary attempt with aiohttp
            async with session.get(initial_url, allow_redirects=True, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    final_url = str(response.url)

                    # Extract content based on what's requested
                    if download_text or download_title:
                        # Only extract content if we need text or title
                        extracted_title, extracted_content = extract_content(
                            content, final_url, extract_method
                        )

                        if download_title:
                            title = extracted_title
                        if download_text:
                            extracted_text = extracted_content

                    # Extract favicon if requested
                    if download_favicon:
                        soup = BeautifulSoup(content, "html.parser")
                        favicon_url = default_favicon  # Start with default
                        icon_links = soup.find_all(
                            "link",
                            rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                        )
                        meta_icons = soup.find_all(
                            "meta", attrs={"content": re.compile(r".ico$", re.I)}
                        )
                        icons = icon_links + meta_icons

                        for icon in icons:
                            found_url = icon.get("href") or icon.get("content")
                            if found_url:
                                # Ensure the URL is absolute
                                parsed_found_url = urlparse(found_url)
                                if not parsed_found_url.scheme:
                                    favicon_url = urljoin(
                                        final_url, found_url
                                    )  # Make absolute relative to final URL
                                else:
                                    favicon_url = found_url  # It's already absolute
                                break  # Use the first one found

                else:
                    logging.info(
                        f"aiohttp request failed for {initial_url} with status {response.status}. Trying requests fallback."
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message="Non-200 status",
                    )

        except (TimeoutError, aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            logging.info(
                f"aiohttp request failed for {initial_url}: {e}. Trying requests fallback."
            )
            # Fallback attempt with requests
            try:
                loop = asyncio.get_running_loop()
                headers = session.headers

                # Run requests.get in an executor
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        initial_url, headers=headers, timeout=10, allow_redirects=True
                    ),
                )

                response.raise_for_status()  # Check for HTTP errors

                content = response.text
                final_url = response.url

                # Extract content based on what's requested
                if download_text or download_title:
                    extracted_title, extracted_content = extract_content(
                        content, final_url, extract_method
                    )

                    if download_title:
                        title = extracted_title
                    if download_text:
                        extracted_text = extracted_content

                # Extract favicon if requested
                if download_favicon:
                    soup = BeautifulSoup(content, "html.parser")
                    favicon_url = default_favicon  # Start with default
                    icon_links = soup.find_all(
                        "link",
                        rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                    )
                    meta_icons = soup.find_all(
                        "meta", attrs={"content": re.compile(r".ico$", re.I)}
                    )
                    icons = icon_links + meta_icons

                    for icon in icons:
                        found_url = icon.get("href") or icon.get("content")
                        if found_url:
                            parsed_found_url = urlparse(found_url)
                            if not parsed_found_url.scheme:
                                favicon_url = urljoin(final_url, found_url)
                            else:
                                favicon_url = found_url
                            break

                logging.info(f"Requests fallback succeeded for {initial_url}")

            except requests.exceptions.RequestException as req_e:
                logging.info(f"Requests fallback also failed for {initial_url}: {req_e}")
                title, extracted_text, final_url, favicon_url = None, None, None, None
            except Exception as fallback_e:
                logging.info(
                    f"Error during requests fallback processing for {initial_url}: {fallback_e}"
                )
                title, extracted_text, final_url, favicon_url = None, None, None, None

        except Exception as e:
            logging.info(
                f"Unexpected error during aiohttp processing for {initial_url}: {e}. No fallback attempted."
            )
            title, extracted_text, final_url, favicon_url = None, None, None, None

    # Apply defaults if extraction succeeded but specific components are missing
    if final_url:
        if download_favicon and not favicon_url:
            favicon_url = default_favicon
        if download_title and not title:
            title = fallback_title

    return title, extracted_text, final_url, favicon_url


async def download_content_parallel_async(
    urls: list[str],
    download_text: bool = True,
    download_title: bool = True,
    download_favicon: bool = True,
    extract_method: ExtractMethod = ExtractMethod.NEWSPAPER,
) -> list[tuple[str, str, str, str]]:
    """
    Download content selectively from a list of URLs in parallel.

    Args:
        urls (List[str]): The list of URLs to download from.
        download_text (bool): Whether to extract text content
        download_title (bool): Whether to extract title
        download_favicon (bool): Whether to extract favicon
        extract_method (ExtractMethod): Method to use for content extraction

    Returns:
        List[Tuple[str, str, str, str]]: List of (title, extracted_text, final_url, favicon_url) tuples.
                                        None for disabled options.
    """
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    # Define headers for aiohttp session
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = {
            url: asyncio.create_task(
                download_content_async(
                    url,
                    session,
                    semaphore,
                    download_text,
                    download_title,
                    download_favicon,
                    extract_method,
                )
            )
            for url in urls
        }

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        # Reorder results based on input URL order
        results = []
        for url in urls:
            result = await tasks[url]
            results.append(result)

    return results


def download_content(
    urls: list[str],
    download_text: bool = True,
    download_title: bool = True,
    download_favicon: bool = True,
    extract_method: ExtractMethod = ExtractMethod.NEWSPAPER,
) -> list[tuple[str, str, str, str]]:
    """
    Download content selectively from a list of URLs in parallel.

    Args:
        urls (List[str]): The list of URLs to download from.
        download_text (bool): Whether to extract text content (default: True)
        download_title (bool): Whether to extract title (default: True)
        download_favicon (bool): Whether to extract favicon (default: True)
        extract_method (ExtractMethod): Method to use for content extraction (default: NEWSPAPER)

    Returns:
        List[Tuple[str, str, str, str]]: List of (title, extracted_text, final_url, favicon_url) tuples.
                                        Returns None for disabled options.
    """
    return run_async_in_thread(
        download_content_parallel_async(
            urls, download_text, download_title, download_favicon, extract_method
        )
    )


if __name__ == "__main__":
    import time

    urls = [
        "https://elchapuzasinformatico.com/2024/08/coolpc-gamer-xi-powered-by-msi/",
        "https://elchapuzasinformatico.com/2024/08/gafas-xr-samsung-soc-snapdragon-xr2-gen-2/",
        "https://www.elcorreo.com/sociedad/educacion/red-publica-25000-profesores-doble-concertada-similares-20240828010340-nt.html",
        "https://elpais.com/economia/2024-08-28/el-cni-alerto-de-la-conexion-rusa-del-grupo-hungaro-que-puja-por-talgo.html",
        "https://www.elmundo.es/espana/2024/08/27/66ce074be4d4d80d358b4582.html",
        "https://www.pccomponentes.com/mejores-moviles-segun-camara?srsltid=AfmBOoo-hnXbFUdnCSGE09qyfhdqnZG0LQnJ_D86_XNUdeuKNmTcI3Cu",
        "https://www.marca.com/futbol/barcelona/2024/08/28/66ce528d22601d0c488b4583.html",
        "https://www.sport.es/es/noticias/barca/promesa-nico-williams-le-aleja-106197709",
    ] * 2  # Reduced multiplier for faster testing

    # Test sample URL for quality check
    sample_url = urls[-2]

    print("=" * 80)
    print("CONTENT EXTRACTION QUALITY COMPARISON")
    print("=" * 80)

    # Get content with all methods for a single URL to compare
    for method in ExtractMethod:
        print(f"\nTesting {method.value.upper()} extraction on sample URL:")
        start_time = time.time()
        sample_results = download_content([sample_url], method)
        method_time = time.time() - start_time

        if sample_results[0]:
            title, text, url, favicon = sample_results[0]
            print(f"Time: {method_time:.2f} seconds")
            print(f"Title: {title}")
            print(f"Text length: {len(text)} characters")
            print(f"Text preview: {text}\n")
        else:
            print(f"Failed to extract content with {method.value}\n")

    print("=" * 80)
    print("PERFORMANCE BENCHMARK FOR ALL METHODS")
    print("=" * 80)

    # Test all extraction methods for performance
    timing_results = {}
    for method in ExtractMethod:
        print(f"\nBenchmarking {method.value}...")

        # Test text extraction only
        start_time = time.time()
        results = download_content(urls, method, download_favicon=False)
        text_only_time = time.time() - start_time

        # Count valid results
        valid_results = sum(1 for r in results if r[0] is not None)

        # Test combined extraction (text and favicon)
        start_time = time.time()
        combined_results = download_content(urls, method)
        combined_time = time.time() - start_time

        # Count valid combined results
        valid_combined = sum(1 for r in combined_results if r[0] is not None)

        timing_results[method.value] = {
            "text_only_time": text_only_time,
            "combined_time": combined_time,
            "valid_results": valid_results,
            "valid_combined": valid_combined,
        }

        print(f"Text only time: {text_only_time:.2f} seconds ({valid_results}/{len(urls)} valid)")
        print(f"Combined time: {combined_time:.2f} seconds ({valid_combined}/{len(urls)} valid)")

    # Dowload only favicon time
    start_time = time.time()
    results = download_content(
        urls, download_favicon=True, download_text=False, download_title=False
    )
    favicon_only_time = time.time() - start_time

    print(f"Favicons: {[x[-1] for x in results]}")

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'Method':<15} {'Text Time (s)':<15} {'Combined Time (s)':<20} {'Success Rate':<15}")
    print("-" * 80)

    for method, results in timing_results.items():
        print(
            f"{method:<15} {results['text_only_time']:<15.2f} {results['combined_time']:<20.2f} "
            f"{results['valid_results']}/{len(urls)} ({results['valid_results'] / len(urls) * 100:.0f}%)"
        )

    # Find fastest method for text extraction
    fastest_text = min(timing_results.items(), key=lambda x: x[1]["text_only_time"])
    # Find fastest method for combined extraction
    fastest_combined = min(timing_results.items(), key=lambda x: x[1]["combined_time"])

    print("\n" + "=" * 30)
    print(f"Fastest text extraction: {fastest_text[0]} ({fastest_text[1]['text_only_time']:.2f}s)")
    print(
        f"Fastest combined extraction: {fastest_combined[0]} ({fastest_combined[1]['combined_time']:.2f}s)"
    )
    print(f"Favicon only time: {favicon_only_time:.2f} seconds")
