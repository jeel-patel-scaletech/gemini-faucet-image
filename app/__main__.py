import os
from google import genai
from typing import TypedDict
from google.genai.types import CreateCachedContentConfig, HttpOptions, Content, Part
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.0-flash-lite"

client = genai.Client()


def read_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as fp:
        return fp.read()


content_from_files = [
    Content(
        role="user",
        parts = [
          Part.from_text(text = file_path),
          Part.from_bytes(data=read_bytes(f"input_images/{file_path}"), mime_type="image/png")
        ]
    )
    for file_path in os.listdir("input_images")
]

# cached_dataset = client.caches.create(
#     model = MODEL_NAME,
#     config = CreateCachedContentConfig(
#         ttl = f"{24 * 60 * 60}s",
#         display_name = "Faucet Images Dataset Cache",
#         contents = content_from_files,
#         system_instruction = (
#           "You are an expert plumbing inventory assistant. "
#           "You have access to a specific catalog of faucet images (provided in context). "
#           "Your job is to visually compare a user-provided photo against this catalog "
#           "and identify the specific catalog item that matches best."
#         )
#     )
# )

print(content_from_files)
# print(cached_dataset)


class FaucetMatch(TypedDict):
    best_match_filename: str
    confidence_level: str
    reasoning: str
    is_exact_match: bool


with open("query_image/fancy.png", "rb") as query:
    QUERY_IMAGE_BYTES = query.read()

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[
        Content(
            role="user",
            parts=[
                Part.from_text(
                    text=(
                        "You are an expert plumbing inventory assistant. "
                        "You have access to a specific catalog of faucet images (provided in context) with filenames. "
                        "Your job is to visually compare a user-provided photo against this catalog "
                        "and identify the specific catalog item that matches best."
                    )
                ),
            ],
        ),
        *content_from_files,
        Content(
            role="user",
            parts=[
                Part.from_text(
                    text="Find the best match for this given faucet from the previously uploaded set of images. Make sure to provide the file name."
                ),
                Part.from_bytes(data=QUERY_IMAGE_BYTES, mime_type="image/png"),
            ],
        ),
    ],
)

print("\n--- API RESPONSE (RAW) ---")
print(response)
print("\n--- API RESPONSE (JSON) ---")
print(response.text)

# # cache.delete()
