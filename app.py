import hashlib
import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS
from gtts.tts import gTTSError
from streamlit_mic_recorder import mic_recorder

from database import get_calls, initialize_database, save_call

load_dotenv()
BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Sankar AI Sales Bot",
    page_icon="🎙️",
    layout="wide",
)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
VALID_STATUSES = [
    "Converted",
    "Interested",
    "Follow-up Required",
    "Not Interested",
    "Demo Requested",
    "No Response",
]
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Urdu": "ur",
}

if not API_KEY:
    st.error("Gemini API key is missing. Create a .env file using .env.example.")
    st.stop()

client = genai.Client(api_key=API_KEY)
initialize_database()

DEFAULTS = {
    "active": False,
    "started_at": None,
    "started_epoch": None,
    "call_id": None,
    "customer_name": "Guest Customer",
    "messages": [],
    "audio_files": [],
    "last_audio_hash": None,
    "pending_transcription_path": None,
    "transcription_error": None,
    "pending_reply": False,
    "reply_error": None,
    "saved_call": False,
    "consent_revision": 0,
    "end_call_error": None,
    "voice_error": None,
    "summary_data": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def ask_ai(prompt, *, json_response=False):
    config = (
        types.GenerateContentConfig(response_mime_type="application/json")
        if json_response
        else None
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    answer = getattr(response, "text", None)
    if not answer:
        raise RuntimeError("The AI returned an empty response. Check model access and API quota.")
    return answer.strip()


def transcribe_audio(audio_bytes):
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            "Transcribe everything the customer says, in the language spoken. "
            "Return only the words spoken. If speech cannot be understood, "
            "return [unclear speech].",
        ],
    )
    return (response.text or "").strip()


def generate_sales_reply():
    history = "\n".join(
        f'{message["role"]}: {message["text"]}'
        for message in st.session_state.messages
    )
    product_details = st.session_state.product_details.strip()
    product_prompt = (
        product_details
        or "No product details provided. Ask the operator for details instead of guessing."
    )
    return ask_ai(
        "You are Sankar, a friendly, professional, and ethical AI sales "
        "representative. Understand the customer's needs, answer their "
        "questions, explain the product using only the supplied product "
        "details, and help them decide whether to buy or book a demo. Never "
        "invent product features, pricing, or promises. If the details do not "
        "answer a question, say so and offer a follow-up. Do not pressure the "
        "customer. Reply in "
        f'{st.session_state.call_language}, using concise natural speech.\n\n'
        f"Product or service details:\n{product_prompt}\n\n"
        f"Conversation so far:\n{history}\n\n"
        "Write only the next response as the sales representative."
    )


def add_audio_recording(speaker, filename):
    relative_path = str(Path(st.session_state.call_id) / filename)
    st.session_state.audio_files.append({
        "speaker": speaker,
        "file": relative_path,
    })
    return RECORDINGS_DIR / relative_path


def make_assistant_message(text):
    message = {"role": "assistant", "text": text}
    audio_buffer = io.BytesIO()
    try:
        gTTS(text=text, lang=LANGUAGES[st.session_state.call_language]).write_to_fp(
            audio_buffer
        )
    except gTTSError as exc:
        st.session_state.voice_error = (
            f"Text is available, but voice playback could not be generated: {exc}"
        )
        return message

    filename = f"assistant_{len(st.session_state.audio_files) + 1:03}.mp3"
    audio_path = add_audio_recording("Assistant", filename)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(audio_buffer.getvalue())
    message["audio"] = audio_buffer.getvalue()
    return message


def add_customer_message(text):
    text = text.strip()
    if not text:
        return

    st.session_state.messages.append({"role": "customer", "text": text})
    generate_assistant_reply()


def generate_assistant_reply():
    try:
        reply = generate_sales_reply()
    except genai.errors.APIError as exc:
        st.session_state.pending_reply = True
        st.session_state.reply_error = format_gemini_error("AI reply", exc)
        return
    except RuntimeError as exc:
        st.session_state.pending_reply = True
        st.session_state.reply_error = f"Could not generate the AI reply: {exc}"
        return

    st.session_state.messages.append(make_assistant_message(reply))
    st.session_state.pending_reply = False
    st.session_state.reply_error = None


def format_gemini_error(action, error):
    if error.code == 503:
        return (
            f"Gemini is temporarily unavailable while {action}. Your data is "
            "preserved. Wait a moment, then use Retry."
        )
    if error.code == 429:
        return (
            f"Gemini rate limit reached while {action}. Your data is preserved; "
            "wait before retrying."
        )
    return f"Gemini could not complete {action} (error {error.code}): {error}"


def transcribe_saved_recording(recording_path):
    try:
        customer_text = transcribe_audio(recording_path.read_bytes())
    except genai.errors.APIError as exc:
        st.session_state.transcription_error = format_gemini_error(
            "transcribing the recording",
            exc,
        )
        return
    except RuntimeError as exc:
        st.session_state.transcription_error = (
            f"Could not transcribe the recording: {exc}"
        )
        return

    if not customer_text or customer_text == "[unclear speech]":
        st.session_state.transcription_error = (
            "Gemini could not understand that recording. You can retry "
            "transcription or type the customer's response."
        )
        return

    st.session_state.pending_transcription_path = None
    st.session_state.transcription_error = None
    add_customer_message(customer_text)


def generate_call_summary():
    customer_messages = [
        message for message in st.session_state.messages
        if message["role"] == "customer"
    ]
    if not customer_messages:
        customer_audio_received = any(
            recording["speaker"] == "Customer"
            for recording in st.session_state.audio_files
        )
        if customer_audio_received:
            return {
                "summary": "Customer audio was received but could not be understood.",
                "requirements": "Not identified because the response was unclear.",
                "objections": "Not identified because the response was unclear.",
                "next_steps": "Follow up with the customer to clarify their response.",
                "status": "Follow-up Required",
            }
        return {
            "summary": "No customer response was received during the call.",
            "requirements": "None identified.",
            "objections": "None identified.",
            "next_steps": "Follow up with the customer at a suitable time.",
            "status": "No Response",
        }

    transcript = "\n".join(
        f'{message["role"].title()}: {message["text"]}'
        for message in st.session_state.messages
    )
    prompt = (
        "Analyze this sales conversation. Return one JSON object with exactly "
        "these string keys: summary, requirements, objections, next_steps, "
        "status. Keep the summary short. Use 'None identified.' when no "
        "requirements or objections were mentioned. Choose status only from "
        f"{json.dumps(VALID_STATUSES)}. Determine it from the conversation: "
        "Converted for a clear purchase, Demo Requested for a requested demo, "
        "Not Interested for a clear rejection, Interested for buying interest, "
        "Follow-up Required when further contact is needed, and No Response "
        "when the customer did not respond.\n\n"
        f"Transcript:\n{transcript}"
    )
    result = json.loads(ask_ai(prompt, json_response=True))
    required_fields = ("summary", "requirements", "objections", "next_steps")
    if not isinstance(result, dict):
        raise ValueError("The AI returned an invalid call summary.")
    if any(not isinstance(result.get(field), str) for field in required_fields):
        raise ValueError("The AI returned an incomplete call summary.")
    if result.get("status") not in VALID_STATUSES:
        raise ValueError("The AI returned an invalid sales status.")
    return {field: result[field].strip() for field in (*required_fields, "status")}


def format_duration(seconds):
    minutes, remainder = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{remainder:02}"


def render_live_status():
    duration = int(time.time() - st.session_state.started_epoch)
    left, right = st.columns(2)
    left.metric("Call duration", format_duration(duration))
    right.metric("Call status", "In progress")


st.title("Sankar AI Sales Bot")
st.caption("Multilingual voice conversations and call tracking")

with st.sidebar:
    st.header("Call setup")
    st.text_input("Customer name", key="customer_name")
    st.text_area(
        "Product or service details",
        key="product_details",
        help="Provide accurate features, pricing, and any details the bot can explain.",
    )
    st.selectbox("Conversation language", list(LANGUAGES), key="call_language")

    if not st.session_state.active:
        consent_key = f"recording_consent_{st.session_state.consent_revision}"
        consent = st.checkbox(
            "I have the customer's consent to record, save, and process this call with AI.",
            key=consent_key,
        )
        st.caption(
            "With consent, audio and transcripts are processed by Gemini; voice "
            "replies use Google TTS. Recordings and call history are saved locally."
        )
        if st.button(
            "Start Call",
            type="primary",
            disabled=not consent,
            width="stretch",
        ):
            st.session_state.active = True
            st.session_state.started_at = datetime.now().isoformat(timespec="seconds")
            st.session_state.started_epoch = time.time()
            st.session_state.call_id = uuid.uuid4().hex
            st.session_state.saved_call = False
            st.session_state.end_call_error = None
            st.session_state.voice_error = None
            st.session_state.summary_data = None
            st.session_state.audio_files = []
            st.session_state.last_audio_hash = None
            st.session_state.pending_transcription_path = None
            st.session_state.transcription_error = None
            st.session_state.pending_reply = False
            st.session_state.reply_error = None
            greeting = (
                f"Hello {st.session_state.customer_name}, thank you for your "
                "time. I am Sankar's AI sales assistant. What would you like "
                "to know?"
            )
            st.session_state.messages = [make_assistant_message(greeting)]
            st.rerun()
    else:
        st.success("Recording consent confirmed; call is in progress.")

if st.session_state.voice_error:
    st.warning(st.session_state.voice_error)
    st.session_state.voice_error = None

if st.session_state.end_call_error:
    st.error(st.session_state.end_call_error)
    st.session_state.end_call_error = None

if st.session_state.saved_call:
    st.success("Call saved.")
    if st.session_state.summary_data:
        st.subheader("AI call summary")
        st.write(st.session_state.summary_data["summary"])
        st.caption(f"Final status: {st.session_state.summary_data['status']}")

if st.session_state.active:
    st.subheader(f"Conversation with {st.session_state.customer_name}")
    st.fragment(run_every="1s")(render_live_status)()
    st.caption(
        f"Language: {st.session_state.call_language}. "
        "Speak into the microphone or type a message."
    )

    latest_message_index = len(st.session_state.messages) - 1
    for index, message in enumerate(st.session_state.messages):
        role = "assistant" if message["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(message["text"])
            if "audio" in message:
                st.audio(
                    message["audio"],
                    format="audio/mp3",
                    autoplay=index == latest_message_index,
                )

    if st.session_state.transcription_error:
        st.warning(st.session_state.transcription_error)
        pending_path = st.session_state.pending_transcription_path
        if pending_path:
            recording_path = Path(pending_path)
            if recording_path.is_file():
                st.audio(recording_path.read_bytes(), format="audio/wav")
            if st.button("Retry transcription", key="retry_transcription"):
                with st.spinner("Retrying transcription..."):
                    transcribe_saved_recording(recording_path)
                st.rerun()
        else:
            st.info("The recording is saved in this call's audio history.")

    if st.session_state.reply_error:
        st.warning(st.session_state.reply_error)
        if st.session_state.pending_reply and st.button(
            "Retry AI reply",
            key="retry_ai_reply",
        ):
            with st.spinner("Retrying AI reply..."):
                generate_assistant_reply()
            st.rerun()

    audio = mic_recorder(
        start_prompt="Record customer response",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key=f"customer_recorder_{st.session_state.call_id}",
    )
    if audio:
        audio_bytes = audio["bytes"]
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            audio_path = add_audio_recording(
                "Customer",
                f"customer_{len(st.session_state.audio_files) + 1:03}.wav",
            )
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(audio_bytes)
            st.session_state.pending_transcription_path = str(audio_path)
            with st.spinner("Transcribing and preparing a reply..."):
                transcribe_saved_recording(audio_path)
            st.rerun()

    prompt = st.chat_input("Type a message")
    if prompt:
        with st.spinner("Preparing a reply..."):
            add_customer_message(prompt)
        st.rerun()

    if st.button("End Call", type="secondary", width="stretch"):
        with st.spinner("Generating the call summary and saving the call..."):
            try:
                summary_data = generate_call_summary()
                transcript = "\n".join(
                    f'{message["role"].title()}: {message["text"]}'
                    for message in st.session_state.messages
                )
                save_call({
                    "customer_name": st.session_state.customer_name,
                    "started_at": st.session_state.started_at,
                    "duration_seconds": int(
                        time.time() - st.session_state.started_epoch
                    ),
                    "transcript": transcript,
                    **summary_data,
                    "audio_files": st.session_state.audio_files,
                })
            except (
                genai.errors.APIError,
                json.JSONDecodeError,
                RuntimeError,
                ValueError,
            ) as exc:
                st.session_state.end_call_error = (
                    f"Could not finish and save the call: {exc}. "
                    "The call is still active; try ending it again."
                )
            else:
                st.session_state.summary_data = summary_data
                st.session_state.active = False
                st.session_state.saved_call = True
                st.session_state.consent_revision += 1
                st.session_state.last_audio_hash = None
        st.rerun()
elif not st.session_state.saved_call:
    st.info("Enter the product details, confirm recording consent, then select **Start Call**.")

with st.expander("Call history", expanded=st.session_state.saved_call):
    calls = get_calls()
    if calls:
        for call in calls:
            with st.expander(
                f'{call["customer_name"]} · {call["started_at"]} · {call["status"]}'
            ):
                legacy_summary = not any(
                    call.get(field)
                    for field in ("summary", "requirements", "objections", "next_steps")
                )
                if legacy_summary:
                    st.info(
                        "This call was saved before automatic AI summaries and "
                        "status assignment were added. The status shown is the "
                        "value saved at that time."
                    )
                st.markdown(f"**Duration:** {format_duration(call['duration_seconds'])}")
                st.markdown(f"**Status:** {call['status']}")
                missing_summary = (
                    "Not generated for this call"
                    if legacy_summary
                    else "Not available"
                )
                st.markdown(f"**Summary:** {call['summary'] or missing_summary}")
                st.markdown(
                    f"**Customer requirements:** "
                    f"{call['requirements'] or missing_summary}"
                )
                st.markdown(
                    f"**Objections:** {call['objections'] or missing_summary}"
                )
                st.markdown(
                    f"**Next steps:** {call['next_steps'] or missing_summary}"
                )
                st.markdown("**Transcript**")
                st.text(call["transcript"])

                try:
                    recordings = json.loads(call["audio_files"] or "[]")
                except json.JSONDecodeError:
                    st.error("Recording metadata for this call could not be read.")
                    recordings = []

                if recordings:
                    st.markdown("**Call recordings**")
                    for recording in recordings:
                        if isinstance(recording, str):
                            speaker, relative_path = "Recording", recording
                        else:
                            speaker = recording.get("speaker", "Recording")
                            relative_path = recording.get("file", "")
                        recording_path = (
                            RECORDINGS_DIR / relative_path
                        ).resolve()
                        safe_path = recording_path.is_relative_to(
                            RECORDINGS_DIR.resolve()
                        )
                        if safe_path and recording_path.is_file():
                            st.caption(speaker)
                            st.audio(
                                recording_path.read_bytes(),
                                format=(
                                    "audio/wav"
                                    if recording_path.suffix.lower() == ".wav"
                                    else "audio/mp3"
                                ),
                            )
                        else:
                            st.warning(f"Recording is unavailable: {relative_path}")
                else:
                    st.caption(
                        "No audio recording was saved for this call. "
                        "It cannot be recreated from the transcript."
                    )
    else:
        st.caption("No calls have been saved yet.")
