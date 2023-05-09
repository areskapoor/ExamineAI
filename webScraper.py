import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import tiktoken
import csv
import uuid


def scrapePage(url, data):
    """
    Function for scraping all the text from a page given a specific condition 
    given the url to that condition
    """
    # Fetch the HTML content of the URL
    response = requests.get(url)
    html_content = response.text
    

    # Parse the HTML using Beautiful Soup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Locate the elements with the specified tag and class attributes
    class1 = 'flex-end flex w-full font-lora text-20 font-medium md:text-30 2xl:text-40 text-primary'
    class2 = 'leading-7 -tracking-2 xl:text-xl xl:leading-9'
    class3 = 'font-lora font-medium capitalize'

    elements1 = soup.find_all('div', class_=class1)
    elements2 = soup.find_all('div', class_=class2)
    elements3 = soup.find_all('h1', class_=class3)

    #Make title elem list same length as other lists
    elements3 = [elements3[0] for i in range(len(elements1))]
    
    for headElem,textElem,titleElem in zip(elements1, elements2, elements3):
        header = headElem.get_text(strip=True)
        text = textElem.get_text(strip=True)
        title = titleElem.get_text(strip=True)
        id = str(uuid.uuid4())
        entry = {"id": id, "title": title, "header": header, "text": text, "url": url}
        data.append(entry)
        
    return data
    
    
def extract_links(url, elem, className):
    response = requests.get(url)
    html_content = response.text
    
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = soup.find_all('a', class_=className)

    links = []

    for element in elements:
        href = element.get('href')
        if href:
            links.append(href)

    return links


def scrapeCategory(url, data):
    #scrape text off of category page
    newData = scrapePage(url, data)
    
    links = extract_links(url, "a", "mb-5 flex items-center pl-4")
    
    baseLink = "https://examine.com"
    
    for link in tqdm(links):
        l = baseLink + link
        newData = scrapePage(l,newData)
    
    return newData
        

def scrapeWebsite(url, data):
    links = extract_links(url, "a", "text-primary underline decoration-primary")
    
    baseLink = "https://examine.com"

    for link in tqdm(links):
        l = baseLink + link
        data = scrapeCategory(l,data)
    
    return data
    


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def main():
    data = scrapeWebsite("https://examine.com/categories/", [])
    # data = scrapeCategory("https://examine.com/categories/brain-health/#all-conditions", [])
    totalTokens = 0
    for x in data:
        totalTokens += num_tokens_from_string(x["text"], "cl100k_base")
    avgTokens = totalTokens/len(data)
    print("total Tokens used:", totalTokens)
    print("Avg Tokens:", avgTokens)

    
    with open('data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'url', 'title','header', 'text'])
        writer.writeheader()
        writer.writerows(data)
    


if __name__ == "__main__":
    main()