from typing import Optional
from langchain_core.tools import tool


@tool
def url_to_text(url: str, requests=None) -> Optional[str]:
    """fetches the text from a given url"""
    try:
        with requests.get(url) as resp:
            return resp.text
    except Exception as e:
        print(f"Failed to get text from {url}: {e}")
        return None
