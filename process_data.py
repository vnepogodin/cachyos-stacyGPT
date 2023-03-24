#!/usr/bin/python
#
# This script takes the Replit documentation and creates the embeddings
#

from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from dotenv import load_dotenv
load_dotenv()
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from bs4 import BeautifulSoup

def extract_text_from_html(file_path):
  html_content = ''
  with open(file_path) as f:
      html_content = f.read()
  soup_obj = BeautifulSoup(html_content, features="html.parser")

  # kill all script and style elements
  for script in soup_obj(["script", "style"]):
      script.extract()    # rip it out

  # get text
  text = soup_obj.get_text()

  # break into lines and remove leading and trailing space on each
  lines = (line.strip() for line in text.splitlines())
  # break multi-headlines into a line each
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  # drop blank lines
  text = '\n'.join(chunk for chunk in chunks if chunk)
  return text


# directory to take files from
ps = list(Path("wiki/").glob("**/*.md"))

# Read in the files and store them in a list
data = []
for p in ps:
  with open(p) as f:
    data.append(f.read())

ps = list(Path("cachyos-website/").glob("*.html"))
for p in ps:
  data.append(extract_text_from_html(p))

ps = list(Path("arch-wiki/").glob("**/*.html"))
for p in ps:
  with open(p) as f:
    data.append(f.read())
#for p in ps:
#  data.append(extract_text_from_html(p))

# Split the text into chunks of 2000 characters (because of LLM context limits)
text_splitter = CharacterTextSplitter(chunk_size=2000, separator="\n")
docs = []
for d in data:
  docs.extend(text_splitter.split_text(d))

# Create a vector store from the documents and save it locally
store = FAISS.from_texts(docs, OpenAIEmbeddings())
faiss.write_index(store.index, "wiki.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
  pickle.dump(store, f)

# After running this script, you should see two new files: faiss_store.pkl (vector store) and docs.index
# Now you are ready to run main.py
