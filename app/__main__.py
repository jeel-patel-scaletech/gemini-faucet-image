import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, cast

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
CATALOG_DIR = Path(os.getenv("CATALOG_DIR", "input_images"))
SYSTEM_PROMPT = (
    "You are an expert plumbing inventory assistant. You have access to a specific "
    "catalog of faucet images (provided in context). Your job is to visually compare "
    "a user-provided photo against this catalog and identify the catalog item that matches best."
)
JSON_INSTRUCTION = (
    "Compare the uploaded faucet photo with the catalog and respond with pure JSON "
    "matching this schema (no markdown, comments, backticks, or prose): "
    '{"best_match_filename": "...", "confidence_level": "...", '
    '"reasoning": "...", "is_exact_match": true|false}.'
)

client = genai.Client()


class FaucetMatch(TypedDict):
    best_match_filename: str
    confidence_level: str
    reasoning: str
    is_exact_match: bool


def _read_bytes(file_path: Path) -> bytes:
    return file_path.read_bytes()


def _guess_mime_type(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(file_path.name)
    return mime_type or "application/octet-stream"


@st.cache_resource(show_spinner=False)
def load_catalog_contents() -> tuple[List[Content], List[Dict[str, bytes]]]:
    contents: List[Content] = []
    previews: List[Dict[str, bytes]] = []

    if not CATALOG_DIR.exists():
        raise FileNotFoundError(f"Catalog directory '{CATALOG_DIR}' does not exist.")

    for image_path in sorted(CATALOG_DIR.iterdir()):
        if not image_path.is_file():
            continue

        image_bytes = _read_bytes(image_path)
        mime_type = _guess_mime_type(image_path)
        if not mime_type.startswith("image/"):
            continue
        contents.append(
            Content(
                role="user",
                parts=[
                    Part.from_text(text=image_path.name),
                    Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        )
        previews.append({"name": image_path.name, "bytes": image_bytes})

    if not contents:
        raise FileNotFoundError(f"No images found inside '{CATALOG_DIR}'.")

    return contents, previews


def run_match(
    question: str,
    image_bytes: bytes,
    image_mime: Optional[str],
    catalog_contents: List[Content],
) -> tuple[Optional[FaucetMatch], str]:
    contents = [
        Content(role="user", parts=[Part.from_text(text=SYSTEM_PROMPT)]),
        *catalog_contents,
        Content(
            role="user",
            parts=[
                Part.from_text(text=f"{JSON_INSTRUCTION}\n\nUser question: {question}"),
                Part.from_bytes(data=image_bytes, mime_type=image_mime or "image/png"),
            ],
        ),
    ]

    response = client.models.generate_content(model=MODEL_NAME, contents=contents)
    raw_text = response.text.strip()

    try:
        parsed = cast(FaucetMatch, json.loads(raw_text))
        return parsed, raw_text
    except json.JSONDecodeError:
        return None, raw_text


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def render_sidebar(previews: List[Dict[str, bytes]]) -> None:
    st.sidebar.header("Query Image")
    uploaded_file = st.sidebar.file_uploader("Upload a faucet image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.session_state["query_image"] = uploaded_file.getvalue()
        st.session_state["query_image_name"] = uploaded_file.name
        uploaded_mime = uploaded_file.type or _guess_mime_type(Path(uploaded_file.name))
        st.session_state["query_image_mime"] = uploaded_mime or "image/png"

    if st.session_state.get("query_image"):
        st.sidebar.image(st.session_state["query_image"], caption=st.session_state.get("query_image_name", "Uploaded image"))
    else:
        st.sidebar.warning("Upload a faucet image to begin.")

    with st.sidebar.expander("Catalog preview", expanded=False):
        st.write(f"{len(previews)} catalog images loaded.")
        cols = st.columns(3)
        for idx, preview in enumerate(previews):
            with cols[idx % 3]:
                st.image(preview["bytes"], caption=preview["name"], use_container_width=True)


def render_chat_interface(catalog_contents: List[Content]) -> None:
    for message in st.session_state["messages"]:
        chat = st.chat_message(message["role"])
        if message["role"] == "assistant" and message.get("match"):
            match = message["match"]
            chat.markdown(
                f"**Match:** `{match['best_match_filename']}`\n\n"
                f"- Confidence: {match['confidence_level']}\n"
                f"- Exact match: {match['is_exact_match']}\n\n"
                f"**Reasoning:** {match['reasoning']}"
            )
            if message.get("raw"):
                chat.code(message["raw"], language="json")
        else:
            content = message.get("content")
            if content:
                chat.markdown(content)
            if message.get("raw"):
                chat.code(message["raw"], language="json")

    user_prompt = st.chat_input("Ask about the faucet or request a comparison")
    if not user_prompt:
        return

    if not st.session_state.get("query_image"):
        st.warning("Upload an image in the sidebar before asking a question.")
        return

    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Comparing faucet image..."):
            match, raw_text = run_match(
                question=user_prompt,
                image_bytes=st.session_state["query_image"],
                image_mime=st.session_state.get("query_image_mime", "image/png"),
                catalog_contents=catalog_contents,
            )

        assistant_message = {"role": "assistant", "raw": raw_text}
        if match:
            assistant_message["match"] = match
            st.success(f"Best match: `{match['best_match_filename']}` (confidence: {match['confidence_level']})")
            st.markdown(f"**Reasoning:** {match['reasoning']}")
            st.caption(f"Exact match: {match['is_exact_match']}")
            st.code(json.dumps(match, indent=2))
        else:
            assistant_message["content"] = "I couldn't parse the model response. Please try again or rephrase your request."
            st.error("Could not parse model response as JSON.")
            st.code(raw_text)

        st.session_state["messages"].append(assistant_message)


def main() -> None:
    st.set_page_config(page_title="Faucet Matcher", layout="wide")
    st.title("Faucet Catalog Matcher")
    st.caption("Upload a photo of a faucet and ask questions to find the closest product in your catalog.")

    catalog_contents, previews = load_catalog_contents()
    init_session_state()
    render_sidebar(previews)
    render_chat_interface(catalog_contents)


if __name__ == "__main__":
    main()
