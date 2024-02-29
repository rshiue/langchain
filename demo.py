
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from os import environ
from bs4 import BeautifulSoup

from rich.console import Console

import logging

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
GITHUB_URL_LANGCHAIN = r"https://github.com/langchain-ai/langchain/releases/tag/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo")
# print the log to file, rotating every 1MB
handler = logging.FileHandler("langchain-demo.log", mode="a")
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

def get_changelogs():
    """read a local file `tags.md`. line by line, save as a list"""
    url_tags = []
    with open("tags.md", "r") as f:
        lines = f.readlines()
        import math
        LIMIT = math.inf
        # LIMIT = 10
        count = 0
        for line in lines:
            if count >= LIMIT:
                break
            count += 1
            tag = line.strip()
            url_tags.append(f"{GITHUB_URL_LANGCHAIN}" + tag)

    logger.debug("\n".join(url_tags))
    import requests
    collection = []
    # make dir at `./tag_pages/`
    if "tag_pages" not in os.listdir():
        os.mkdir("tag_pages")

    for url in url_tags:

        # first, check `./tag_pages/` to see if the page has already been downloaded.
        # if it has, load the file 
        # if it hasn't, continue to load the web page
        
        tagname = url.split('/')[-1]
        if f"{tagname}.html" in os.listdir("tag_pages"):

            
            with open(f"tag_pages/{tagname}.html", "r") as f:
                content = f.read()
                # do data cleansing...
                content = BeautifulSoup(content, "html.parser").get_text()
                # remove leading and trailing whitespaces
                content = content.strip()
                # remove duplicated line breaks: `\n`
                content = "\n".join([line for line in content.split("\n") if line.strip() != ""])
                logger.debug(content[:100])
                collection.append(content)
            
            logger.debug(f"[cached]loading content from {url}")
            
        else: 
            
            logger.debug(f"loading content from remote: {url}")
            response = requests.get(url)
            
            # convert html to plain text, using `BeautifulSoup`
            soup = BeautifulSoup(response.text, "html.parser")

            # write the HTML content to a file at `./tag_pages/`
            with open(f"tag_pages/{tagname}.html", "w") as f:
                f.write(soup.prettify())

            plain_text = soup.get_text()
            logger.info(f"plain_text: {plain_text}")
            collection.append(plain_text)

    return collection
    
def main(question, force_rebuild=False):

    logger.info("preparing vectorstore ... ")
    embedding = OpenAIEmbeddings()

    if force_rebuild is False and "faiss.index" in os.listdir():
        from langchain.vectorstores.utils import DistanceStrategy
        logger.info(f"Found existed vectorstore. We'll use this local store.")
        vectorstore = FAISS.load_local( "./faiss.index", embedding, distance_strategy=DistanceStrategy.COSINE)
    else:
        content = get_changelogs()
        logger.debug(f"content: {content}")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=32)
        docs = []
        for item in content:
            docs.extend(text_splitter.split_text(item))
        # docs = [ text_splitter.split_text(item) for item in content ]
        vectorstore = FAISS.from_texts(
            docs, embedding=embedding
        )
        vectorstore.save_local("faiss.index")

    # logger.info(f"retreived_docs: {retreived_docs[0].page_content}")

    retriever = vectorstore.as_retriever()
    
    from langchain import hub

    prompt = hub.pull("rlm/rag-prompt")


    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")


    def format_docs(docs):
        logger.info(f"docs to format : {len(docs)}")
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(question)
    logger.info(result)

    print("\n\n")
    console = Console(width=80)
    console.print(result, style="green on white")
    console.print("\n")

    
if __name__ == "__main__":
    assert OPENAI_API_KEY is not None, "OpenAI API key is not set"
    import sys
    message = sys.argv[1]
    is_force_rebuild = sys.argv[2] if len(sys.argv) > 2 else False
    if len(message) == 0:
        c = Console()
        c.print("Please enter your question. E.g. python demo.py \"What is the v0.0.102 doing?\"", style="bold green")
        sys.exit(1)
    else:
        main(question=message, force_rebuild=is_force_rebuild)

