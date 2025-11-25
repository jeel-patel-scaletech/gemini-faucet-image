import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict, cast

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


class UploadedImage(TypedDict):
    name: str
    bytes: bytes
    mime: str


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
    user_images: List[UploadedImage],
    catalog_contents: List[Content],
) -> tuple[Optional[FaucetMatch], str]:
    if not user_images:
        raise ValueError("At least one user image is required to run a match.")

    attachment_summary = ", ".join(image["name"] for image in user_images)
    contents = [
        Content(role="user", parts=[Part.from_text(text=SYSTEM_PROMPT)]),
        *catalog_contents,
        Content(
            role="user",
            parts=[
                Part.from_text(
                    text=(
                        f"{JSON_INSTRUCTION}\n\nUser question: {question}\n"
                        f"Attached images: {attachment_summary}"
                    )
                ),
                *(
                    part
                    for image in user_images
                    for part in (
                        Part.from_text(text=f"Photo: {image['name']}"),
                        Part.from_bytes(data=image["bytes"], mime_type=image["mime"]),
                    )
                ),
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
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0


def render_sidebar(previews: List[Dict[str, bytes]]) -> None:
    st.sidebar.header("Catalog Preview")
    with st.sidebar.expander("Reference catalog", expanded=False):
        st.write(f"{len(previews)} catalog images loaded.")
        cols = st.columns(3)
        for idx, preview in enumerate(previews):
            with cols[idx % 3]:
                st.image(preview["bytes"], caption=preview["name"], use_container_width=True)
    st.sidebar.info("Attach as many faucet photos as you need directly from the main chat input.")


def _serialize_uploaded_files(uploaded_files: Optional[Sequence[Any]]) -> List[UploadedImage]:
    images: List[UploadedImage] = []
    if not uploaded_files:
        return images

    for uploaded_file in uploaded_files:
        file_name = getattr(uploaded_file, "name", "uploaded-image")
        file_bytes = uploaded_file.getvalue()
        file_mime = getattr(uploaded_file, "type", None) or _guess_mime_type(Path(file_name))
        images.append({"name": file_name, "bytes": file_bytes, "mime": file_mime or "image/png"})

    return images


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
            if message.get("images"):
                for image in message["images"]:
                    chat.image(image["bytes"], caption=image["name"], use_container_width=True)
        else:
            content = message.get("content")
            if content:
                chat.markdown(content)
            if message.get("images"):
                for image in message["images"]:
                    chat.image(image["bytes"], caption=image["name"], use_container_width=True)
            if message.get("raw"):
                chat.code(message["raw"], language="json")

    uploader_key = st.session_state.get("uploader_key", 0)
    uploaded_files = st.file_uploader(
        "Attach faucet photo(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"chat_uploader_{uploader_key}",
    )
    pending_images = _serialize_uploaded_files(uploaded_files)

    if pending_images:
        st.caption(f"{len(pending_images)} image(s) attached for the next message.")
        cols = st.columns(min(3, len(pending_images)))
        for idx, image in enumerate(pending_images):
            with cols[idx % len(cols)]:
                st.image(image["bytes"], caption=image["name"], use_container_width=True)

    user_prompt = st.chat_input("Ask about the faucet or request a comparison")
    if not user_prompt:
        return

    if not pending_images:
        st.warning("Attach at least one faucet image before sending a question.")
        return

    user_message: Dict[str, Any] = {"role": "user", "content": user_prompt}
    if pending_images:
        user_message["images"] = pending_images
    st.session_state["messages"].append(user_message)

    user_chat = st.chat_message("user")
    user_chat.markdown(user_prompt)
    for image in pending_images:
        user_chat.image(image["bytes"], caption=image["name"], use_container_width=True)

    with st.chat_message("assistant"):
        with st.spinner("Comparing faucet image..."):
            match, raw_text = run_match(
                question=user_prompt,
                user_images=pending_images,
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
        st.session_state["uploader_key"] = uploader_key + 1


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
