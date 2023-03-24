#!/usr/bin/python
#
# DEVELOPER : wiz64
#

import faiss
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
import pickle
from dotenv import load_dotenv
load_dotenv()
import os
import discord

# Load the index from disk
index = faiss.read_index("wiki.index")

with open("faiss_store.pkl", "rb") as f:
  # Load the vector store from disk
  store = pickle.load(f)

# Set the index on the vector store
store.index = index
# add this to the string if you don't want irrevalent stuff
#  If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

prompt_template = """You are a girl named Stacy working as CachyOS support staff. Use the following pieces of context to answer the question at the end. You are talking to {asker} through discord

{context}

Question: {question}
Helpful Answer:"""

prompt = Prompt(template=prompt_template,
                input_variables=["context", "question","asker"])

# We keep the temperature at 0 to keep the assistant factual
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0))

# Gateway intents (privileges) for the Discord bot
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
# Confirm that the bot is ready
async def on_ready():
  print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
  # Make sure the bot doesn't reply to itself
  if message.author == client.user:
    return

  # Invoke the bot using the command: !replit
  if message.content.lower().__contains__("stacy"):
    # Get the question from the user
    asker = message.author.name
    question = message.content
    # Run a similarity search on the docs to get the most relevant context
    docs = store.similarity_search(question)
    contexts = []
    for j, doc in enumerate(docs):
      contexts.append(f"Context {j}:\n{doc.page_content}")
    # Use the context to answer the question
    answer = llm_chain.predict(question=question,
                               context="\n\n".join(contexts),asker=asker)
    # Finally, reply directly to the user with an answer
    await message.reply(answer)


# Run the bot (make sure you set your Discord token as an environment variable)
client.run(os.getenv("DISCORD_TOKEN"))
