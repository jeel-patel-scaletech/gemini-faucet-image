from pathlib import Path
from typing import List, Literal, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part

load_dotenv()

MODEL_NAME = "gemini-2.0-flash-lite"
SYSTEM_PROMPT = (
    "You are an expert plumbing inventory assistant. "
    "You have access to a specific catalog of faucet images (provided in context) with filenames. "
    "Your job is to visually compare a user-provided photo against this catalog "
    "and identify the specific catalog item that matches best. "
    "Explain your reasoning correctly about why did you make the decision of choosing a specific product."
)
CATALOG_DIR = "input_images"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

client = genai.Client()


class ChatMessage(TypedDict, total=False):
    role: Literal["user", "assistant"]
    text: str
    image_bytes: bytes
    image_mime: str


def read_bytes(file_path: Path) -> bytes:
    with open(file_path, "rb") as fp:
        return fp.read()


def build_catalog_context(directory: str) -> List[Content]:
    catalog_path = Path(directory)
    if not catalog_path.exists():
        return []

    contents: List[Content] = []
    for file_path in sorted(catalog_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        image_bytes = read_bytes(file_path)
        mime_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
        contents.append(
            Content(
                role="user",
                parts=[
                    Part.from_text(text=file_path.name),
                    Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        )

    return contents


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[ChatMessage] = []
    if "catalog_context" not in st.session_state:
        st.session_state.catalog_context = build_catalog_context(CATALOG_DIR)


def build_conversation_contents(messages: List[ChatMessage]) -> List[Content]:
    conversation_contents: List[Content] = []
    for message in messages:
        parts = []
        text_value = (message.get("text") or "").strip()
        if text_value:
            parts.append(Part.from_text(text=text_value))
        elif message["role"] == "user" and message.get("image_bytes"):
            parts.append(Part.from_text(text="Find the best match for this faucet from the catalog."))

        if message["role"] == "user":
            image_bytes = message.get("image_bytes")
            if image_bytes:
                parts.append(
                    Part.from_bytes(data=image_bytes, mime_type=message.get("image_mime") or "image/png")
                )

        if not parts:
            continue

        conversation_contents.append(Content(role=message["role"], parts=parts))

    return conversation_contents


def call_model(messages: List[ChatMessage]) -> str:
    contents = [
        Content(role="user", parts=[Part.from_text(text=SYSTEM_PROMPT)]),
        *st.session_state.catalog_context,
        *build_conversation_contents(messages),
    ]

    response = client.models.generate_content(model=MODEL_NAME, contents=contents)
    if response.text:
        return response.text.strip()
    return "The model did not return any text."


def render_chat_history(messages: List[ChatMessage]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            display_text = message.get("text") or ("(Image only)" if message.get("image_bytes") else "")
            if display_text:
                st.write(display_text)
            if image_bytes := message.get("image_bytes"):
                st.image(image_bytes, caption="Uploaded faucet", width=320)


def main() -> None:
    st.set_page_config(page_title="Faucet Finder", page_icon="🛠️")
    st.title("Faucet Finder Chat")
    st.caption("Ask the assistant to identify faucets and upload photos when needed.")

    ensure_session_state()

    if not st.session_state.catalog_context:
        st.error(f"No catalog images found in `{CATALOG_DIR}`. Add images to continue.")
        return

    render_chat_history(st.session_state.messages)

    with st.form("chat-input", clear_on_submit=True):
        user_message = st.text_area("Message", placeholder="Describe the faucet or ask for help...", height=80)
        uploaded_file = st.file_uploader(
            "Optional: upload a faucet photo", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        text = user_message.strip()
        if not text and uploaded_file is None:
            st.warning("Please enter a message or upload an image before sending.")
            st.stop()

        image_bytes: Optional[bytes] = None
        image_mime: Optional[str] = None
        if uploaded_file:
            image_bytes = uploaded_file.read()
            image_mime = uploaded_file.type or "image/png"

        st.session_state.messages.append(
            ChatMessage(
                role="user",
                text=text,
                image_bytes=image_bytes if image_bytes else None,
                image_mime=image_mime if image_mime else None,
            )
        )

        with st.spinner("Finding the best match..."):
            try:
                assistant_text = call_model(st.session_state.messages)
            except Exception as exc:
                st.session_state.messages.append(
                    ChatMessage(role="assistant", text=f"Error calling Gemini: {exc}")
                )
            else:
                st.session_state.messages.append(ChatMessage(role="assistant", text=assistant_text))

        st.rerun()


if __name__ == "__main__":
    main()
