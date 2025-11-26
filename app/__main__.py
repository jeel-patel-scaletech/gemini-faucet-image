import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Type

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL_NAME = "gemini-2.0-flash-lite"
SYSTEM_PROMPT = (
    "You are an expert plumbing inventory assistant. "
    "You have access to a specific catalog of faucet images (provided in context) with filenames. "
    "Your job is to visually compare a user-provided photo against this catalog "
    "and identify the specific catalog items that match best. "
    "Always respond with JSON that follows this schema exactly:\n"
    '{\n'
    '  "matches": [\n'
    "    {\n"
    '      "filename": "<catalog filename>",\n'
    '      "title": "<title or empty string>",\n'
    '      "brand": "<brand or empty string>",\n'
    '      "color": "<color or empty string>",\n'
    '      "confidence": <number between 0 and 1>,\n'
    '      "reasoning": "<short explanation>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Return exactly three entries in the matches array (sorted by confidence, "
    "highest first) and only reference filenames that exist in the provided catalog. "
    "Do not output any explanation outside of this JSON."
)
CATALOG_DIR = "input_images"
IMAGE_METADATA_FILE = "image_metadata.json"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

client = genai.Client()


class ChatMessage(TypedDict, total=False):
    role: Literal["user", "assistant"]
    text: str
    image_bytes: bytes
    image_mime: str
    matches: List[Dict[str, Any]]


class FaucetMatchModel(BaseModel):
    filename: str
    title: Optional[str] = ""
    brand: Optional[str] = ""
    color: Optional[str] = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class FaucetResponseModel(BaseModel):
    matches: List[FaucetMatchModel]


def _validate_model(model_cls: Type[BaseModel], data: Any) -> BaseModel:
    validator = getattr(model_cls, "model_validate", None)
    if callable(validator):
        return validator(data)
    parse_obj = getattr(model_cls, "parse_obj")
    return parse_obj(data)


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_model_response(response_text: str) -> FaucetResponseModel:
    print(response_text)
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response was not valid JSON.") from exc

    parsed = _validate_model(FaucetResponseModel, data)
    matches = parsed.matches
    # if len(matches) != 3:
        # raise ValueError("Model response must contain exactly three matches.")
    return parsed


def read_bytes(file_path: Path) -> bytes:
    with open(file_path, "rb") as fp:
        return fp.read()


def load_image_metadata(metadata_file: str) -> Dict[str, Any]:
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as fp:
            k = json.load(fp)
            for key, entry in k.items():
                k[key] = {
                    'Title' : entry['Title'],
                    'Brand' : entry['Brand'],
                    'VarDim_Color' : entry['VarDim_Color'],
                }
            return k
    except json.JSONDecodeError:
        return {}


def format_metadata(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return "No additional metadata provided."
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ": "))


def build_catalog_context(directory: str, metadata_map: Dict[str, Any]) -> List[Content]:
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
        metadata_text = format_metadata(metadata_map.get(file_path.name, {}))

        contents.append(
            Content(
                role="user",
                parts=[
                    Part.from_text(text=f"{file_path.name}\nMetadata: {metadata_text}"),
                    Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        )

    return contents


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[ChatMessage] = []
    if "catalog_context" not in st.session_state:
        st.session_state.image_metadata = load_image_metadata(IMAGE_METADATA_FILE)
        st.session_state.catalog_context = build_catalog_context(CATALOG_DIR, st.session_state.image_metadata)


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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )
    if not response.text:
        return "The model did not return any text."
    resp_text = response.text.strip()
    resp_text = resp_text.removeprefix('```json')
    resp_text = resp_text.removeprefix('```')
    resp_text = resp_text.removesuffix('```')

    return resp_text


def render_chat_history(messages: List[ChatMessage]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            matches = message.get("matches") or []
            display_text = message.get("text") or ("(Image only)" if message.get("image_bytes") else "")
            if display_text and not matches:
                st.write(display_text)
            if image_bytes := message.get("image_bytes"):
                st.image(image_bytes, caption="Uploaded faucet", width=320)
            if matches:
                st.json({"matches": matches})
                for idx, match in enumerate(matches, start=1):
                    filename = match.get("filename", "Unknown file")
                    confidence = match.get("confidence")
                    reasoning = match.get("reasoning", "")
                    st.markdown(f"**Match {idx}: {filename} (confidence: {confidence:.2f})**" if isinstance(confidence, (int, float)) else f"**Match {idx}: {filename}**")
                    if reasoning:
                        st.write(reasoning)
                    metadata = st.session_state.image_metadata.get(filename, {})
                    if metadata:
                        st.caption(f"Title: {metadata.get('Title', 'N/A')} | Brand: {metadata.get('Brand', 'N/A')} | Color: {metadata.get('VarDim_Color', 'N/A')}")
                    image_path = Path(CATALOG_DIR) / filename
                    if image_path.exists():
                        st.image(str(image_path), caption=f"Catalog image: {filename}", width=320)
                    else:
                        st.warning(f"Catalog image `{filename}` was not found.")


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
                try:
                    parsed_response = parse_model_response(assistant_text)
                except (ValidationError, ValueError) as exc:
                    st.session_state.messages.append(
                        ChatMessage(
                            role="assistant",
                            text=(
                                "Failed to decode the model response as valid faucet matches. "
                                f"Please try again. Details: {exc}"
                            ),
                        )
                    )
                else:
                    response_dict = _model_to_dict(parsed_response)
                    st.session_state.messages.append(
                        ChatMessage(
                            role="assistant",
                            text=json.dumps(response_dict, indent=2),
                            matches=[_model_to_dict(match) for match in parsed_response.matches],
                        )
                    )

        st.rerun()


if __name__ == "__main__":
    main()
