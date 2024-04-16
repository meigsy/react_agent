import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def url_to_text(url):
    """Fetches the content of a URL and returns the text content."""
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the parsed HTML
        text = soup.get_text(separator=' ', strip=True)

        return text[:128000]
    except requests.RequestException as e:
        return f"An error occurred while fetching the URL: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    print(url_to_text.invoke({"url": "https://en.wikipedia.org/wiki/France"}))
