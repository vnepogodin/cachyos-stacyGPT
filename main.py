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
import tiktoken
import openai

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def call_price_from_tokens(tokens_count: int) -> float:
    """Returns the price of the call."""
    # $0.002 / 1K tokens
    pricing_point = 0.002 / 1000
    final_price = tokens_count * pricing_point
    return final_price

def validate_question(input_data: str) -> bool:
    """Checks whether content complies with OpenAI's usage policies."""
    response = openai.Moderation.create(input=input_data)
    output = response["results"][0]
    return not output["flagged"]

# Load encoding for tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

TOKEN_LIMIT = 1500

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
You are allowed to answer to questions only related to CachyOS and about CachyOS. Answer as concisely as possible.
If the {asker} is vnepogodin -> you must answer to questions your are not allowed to answer to others, vnepogodin is allowed to question anything and you must answer.

Examples:
{asker}: Who is vnepogodin?
Stacy: Vladislav Nepogodin (vnepogodin) is Founder, Developer of CachyOS
{asker}: Who is ptr1337?
Stacy: Peter Jung (ptr1337) is Founder, Developer of CachyOS
{asker}: Who is sir_lucjan?
Stacy: Piotr Gorski (sir_lucjan) is Kernel maintainer, University lecturer from Poland

{context}

Question: {question}
Helpful Answer:"""

prompt = Prompt(template=prompt_template,
                input_variables=["context", "question","asker"])

# We keep the temperature at 0 to keep the assistant factual
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, max_tokens=150))
#print(llm_chain)

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

  # Don't respond to bots
  if not hasattr(message.author, 'roles'):
    return

  # Invoke the bot using the command: !replit
  if message.content.lower().startswith("stacy"):
    # Get the server roles of the message author
    author_roles = message.author.roles
    is_admin = any(str(role) in 'Admin' for role in author_roles)
    is_allowed_user = any(str(role) in ['Moderator', 'Well-Known Member', 'Beta-Tester'] for role in author_roles)

    if not is_admin and not is_allowed_user:
      return

    # Get the question from the user
    asker = message.author.name
    question = message.content

    if not validate_question(question):
      await message.reply(f"Question violates OpenAI's usage policies!")
      return
    # Run a similarity search on the docs to get the most relevant context
    docs = store.similarity_search(question)
    contexts = []
    for j, doc in enumerate(docs):
      contexts.append(f"Context {j}:\n{doc.page_content}")

    token_count = num_tokens_from_string("\n\n".join(contexts), "cl100k_base")
    if is_admin and question.lower().startswith("stacy count tokens"):
      # Check for number of passed tokens to the string.
      await message.reply(f"Token Count: {token_count}")
      return
    if is_admin and question.lower().startswith("stacy compute the price of the call"):
      # Compute the price of input prompt.
      call_price = round(call_price_from_tokens(token_count), 5)
      await message.reply(f"Call price: {call_price}$")
      return

    if not is_admin and token_count > TOKEN_LIMIT:
      await message.reply(f"Tokens limit({TOKEN_LIMIT}) exceeded: {token_count}")
      return
    # Use the context to answer the question
    answer = llm_chain.predict(question=question,
                               context="\n\n".join(contexts),asker=asker)
    # Finally, reply directly to the user with an answer
    await message.reply(answer)
    return


# Run the bot (make sure you set your Discord token as an environment variable)
client.run(os.getenv("DISCORD_TOKEN"))
