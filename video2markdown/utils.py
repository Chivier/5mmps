import os
import httpx
from openai import OpenAI


def openai_simple_agent(model="gpt-4o", messages=None):
    # Create a new OpenAI client
    if messages is None:
        messages = [{"role": "system", "content": "You are a helpful assistent, help me finish my work."},
                    {"role": "user", "content": "Hello"}]
    proxy_url = os.environ.get("OPENAI_PROXY_URL")
    client = OpenAI() if proxy_url is None or proxy_url == "" else OpenAI(http_client=httpx.Client(proxy=proxy_url))
    # Define the model and prompt
    completion = client.chat.completions.create(model=model, messages=messages)
    print(completion.choices[0].message.content)


def openai_read_image(base64_image):
    proxy_url = os.environ.get("OPENAI_PROXY_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if proxy_url is None or proxy_url == "":
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, http_client=httpx.Client(proxy=proxy_url))
    MODEL = "gpt-4o"
    response = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me read this "
                                      "image. This is an instructions book."}, # system message
        {"role": "user", "content": [
            {"type": "text", "text": "Read all the text in this image. If there are some images, describe them. "
                                     "Give me the text and image description in Markdown format."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ], temperature=0.0, )


if __name__ == "__main__":
    pass