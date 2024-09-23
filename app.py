import os
import chainlit as cl
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client

@cl.on_message
async def on_message(message: cl.Message):
    # Define model settings
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.5,
    }

    # Make a request to OpenAI
    response = await aclient.chat.completions.create(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message.content}
    ],
    **settings)

    # Send response back to Chainlit
    await cl.Message(content=response.choices[0].message.content).send()
