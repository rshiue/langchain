
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from os import environ
from bs4 import BeautifulSoup

import logging

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
GITHUB_URL_LANGCHAIN = r"https://github.com/langchain-ai/langchain/releases/tag/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LangChain Demo")

def get_changelogs():
    """read a local file `tags.md`. line by line, save as a list"""
    url_tags = []
    with open("tags.md", "r") as f:
        lines = f.readlines()
        import math
        # LIMIT = math.inf
        LIMIT = 5
        count = 0
        for line in lines:
            if count >= LIMIT:
                break
            count += 1
            tag = line.strip()
            url_tags.append(f"{GITHUB_URL_LANGCHAIN}" + tag)

    logger.info(url_tags)
    import requests
    collection = []
    # make dir at `./tag_pages/`
    if "tag_pages" not in os.listdir():
        os.mkdir("tag_pages")

    for url in url_tags:

        # first, check `./tag_pages/` to see if the page has already been downloaded.
        # if it has, load the file 
        # if it hasn't, continue to load the web page

        if url.split('/')[-1] in os.listdir("tag_pages"):

            with open(f"tag_pages/{url.split('/')[-1]}.html", "r") as f:
                content = f.read()
                collection.append(content)
                continue

        # use `request` to GET the remote web page.
        # fire GET 
        logger.info(f"loading content from {url}")
        response = requests.get(url)
        
        # convert html to plain text, using `BeautifulSoup`
        soup = BeautifulSoup(response.text, "html.parser")

        # write the HTML content to a file at `./tag_pages/`
        with open(f"tag_pages/{url.split('/')[-1]}.html", "w") as f:
            f.write(soup.prettify())

        # loader = BSHTMLLoader(soup.get_text())
        # loader = UnstructuredMarkdownLoader(soup.get_text())
        # loader.load()
        collection.append(soup.get_text())

    return collection
    


def retrieve():
    loader = UnstructuredHTMLLoader()

def get_promps():
    pass

def llm():
    pass


def temp():

    content = get_changelogs()

    logger.info(f"content: {content}")

    vectorstore = FAISS.from_texts(
        content, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")

    chain = ( 
            { "context": retriever, "question": RunnablePassthrough()}
            | prompt 
            | model
            | StrOutputParser()
             
             )
    
    QUESTION = """what is the major changes between theses versions?
"""
    result = chain.invoke(QUESTION)
    logger.info(result)



def main():
    logger.info(f"default key {OPENAI_API_KEY}")
    # urls = get_changelogs()
    # logger.info(f"found tags:\n{urls}")
    temp()



if __name__ == "__main__":
    main()