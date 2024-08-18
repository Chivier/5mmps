import base64
import os

import openai
from openai import OpenAI


def simple_openai_client():
    proxy_url = os.environ.get("OPENAI_PROXY_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if proxy_url is None or proxy_url == "":
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url=f"{proxy_url}/v1")
    print(client)
    return client


def openai_merge_info(info1, info2):
    MODEL = "gpt-4-turbo-preview"
    client = simple_openai_client()

    request_info = (f"Here are 2 paragraphs from an instruction book. I want to merge them into one. \n"
                    f"<PARAGRAPH1>\n"
                    f"{info1}\n"
                    f"</PARAGRAPH1>\n"
                    f"<PARAGRAPH2>\n"
                    f"{info2}\n"
                    f"</PARAGRAPH2>\n"
                    f"First paragraph is more reliable than the second one. If there are any contradictions, please "
                    f"follow the first paragraph. \n"
                    f"Result: ")

    response = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me merge these two "
                                      "paragraphs into one."},
        {"role": "user", "content": [{"type": "text", "text": request_info}]}])

    return response["choices"][0]["message"]["content"]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def openai_read_image(image_path):
    MODEL = "gpt-4o"
    # Get the client
    client = simple_openai_client()

    base64_image = encode_image(image_path)
    # Encode the image
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me read this"
                                              " image. This is an instructions book."},  # system message
                {"role": "user", "content": [
                    {"type": "text", "text": "Read all the text in this image. If there are some images, describe them."
                                             " Give me the text and image description in Markdown format. Only give me "
                                             "the final result only, do not translate or give me useless content. Just "
                                             "Markdown result: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
            ],
            temperature=0.3, max_tokens=400)

        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    pass
