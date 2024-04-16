from typing import Optional

import requests
from langchain_core.tools import tool


@tool
def url_to_text(url: str) -> Optional[str]:
    """fetches the text from a given url"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept": "text/html"
    }

    with requests.get(url,
                      headers=headers,
                      verify=False,
                      timeout=10) as resp:
        return resp.text


if __name__ == "__main__":
    print(url_to_text.invoke({"url": "https://en.wikipedia.org/wiki/France"}))
