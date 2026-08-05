import os
import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as sp
from torch.amp import autocast

from model_def import behzadGPT, CONFIG

# --- Configuration ---
MODEL_CHECKPOINT = os.environ.get("BEHZADGPT_CHECKPOINT", "model/model.pth")
TOKENIZER_PATH = os.environ.get("BEHZADGPT_TOKENIZER", "tokenizer/unigram_8k.model")
EOT_STRING = "<EOT>"
FALLBACK_EOT_ID = 3
MAX_NEW_TOKENS = 300
TOP_K = 4

# Force CPU for Streamlit Free Tier to stay within the 1GB RAM limit
device_type = "cpu"
device = torch.device(device_type)

st.set_page_config(page_title="behzadGPT", page_icon="🤖")

# --- Model Loading (Cached so it only loads once per server) ---
@st.cache_resource(show_spinner="Loading model weights into memory...")
def load_model():
    if not os.path.exists(MODEL_CHECKPOINT):
        st.error(f"Checkpoint not found at {MODEL_CHECKPOINT}.")
        st.stop()
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer not found at {TOKENIZER_PATH}.")
        st.stop()

    m = behzadGPT(CONFIG)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.to(device)
    m.eval()

    tok = sp.SentencePieceProcessor()
    tok.load(TOKENIZER_PATH)

    piece_id = tok.piece_to_id(EOT_STRING)
    resolved_eot_id = piece_id if piece_id not in (0, tok.unk_id()) else FALLBACK_EOT_ID

    return m, tok, resolved_eot_id

model, tokenizer, eot_id = load_model()

# --- Generation Logic ---
@torch.no_grad()
def generate(context_text: str, bot_name: str) -> tuple[str, str]:
    token_id = tokenizer.encode_as_ids(context_text[-512:])

    for _ in range(MAX_NEW_TOKENS):
        window = token_id[-CONFIG["context_length"]:]
        tokens = torch.tensor(window, dtype=torch.int64).unsqueeze(0).to(device)

        with autocast(device_type=device_type, enabled=False): # AMP disabled for CPU
            output = model(tokens)

        logits = output[0, -1]
        top_k_values, top_k_indices = torch.topk(logits, k=TOP_K, dim=-1)
        probs = F.softmax(top_k_values, dim=-1)
        sampled_relative_idx = torch.multinomial(probs, num_samples=1).item()
        next_id = top_k_indices[sampled_relative_idx].item()
        token_id.append(next_id)
        if next_id == eot_id:
            break

    decoded = tokenizer.decode(token_id)
    marker = f"{bot_name}:"
    idx = decoded.rfind(marker)
    reply = decoded[idx + len(marker):] if idx != -1 else decoded
    reply = reply.split(EOT_STRING)[0].strip()
    return reply, decoded

# --- Session State Management ---
if "active_session" not in st.session_state:
    st.session_state.active_session = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""
if "bot_name" not in st.session_state:
    st.session_state.bot_name = "Mario"
if "user_name" not in st.session_state:
    st.session_state.user_name = "Behzad"

# --- UI: Setup Screen ---
if not st.session_state.active_session:
    st.title("New session")
    st.markdown("Set the scene before you start talking. This becomes the model's system prompt. Keep it simple and within 3 lines.")
    
    sys_prompt = st.text_area("Roleplay / system prompt", "Mario is a very good friend. He always helps Behzad and loves to talk to him. Today Mario met Behzad.")
    
    col1, col2 = st.columns(2)
    with col1:
        u_name = st.text_input("Your name", "Behzad")
    with col2:
        b_name = st.text_input("Bot's name", "Mario")
        
    if st.button("Start chat", use_container_width=True):
        st.session_state.user_name = u_name.strip() or "You"
        st.session_state.bot_name = b_name.strip() or "Bot"
        
        opening_context = f"System: {sys_prompt.strip()} {EOT_STRING} {st.session_state.bot_name}: "
        
        with st.spinner("Initializing session..."):
            reply, decoded = generate(opening_context, st.session_state.bot_name)
        
        st.session_state.context = decoded
        st.session_state.messages.append({"role": "assistant", "name": st.session_state.bot_name, "content": reply})
        st.session_state.active_session = True
        st.rerun()

# --- UI: Chat Screen ---
else:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"Chatting with {st.session_state.bot_name}")
    with col2:
        if st.button("New Session"):
            st.session_state.active_session = False
            st.session_state.messages = []
            st.session_state.context = ""
            st.rerun()
            
    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f"**{msg['name']}**\n\n{msg['content']}")

    # Chat Input
    if prompt := st.chat_input("Type a message..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "name": st.session_state.user_name, "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.user_name}**\n\n{prompt}")

        # Generate bot response
        full_prompt = f"{st.session_state.context} {st.session_state.user_name}: {prompt.strip()} {EOT_STRING} {st.session_state.bot_name}: "
        
        with st.chat_message("assistant"):
            with st.spinner("Typing..."):
                reply, decoded = generate(full_prompt, st.session_state.bot_name)
            st.markdown(f"**{st.session_state.bot_name}**\n\n{reply}")
            
        st.session_state.context = decoded
        st.session_state.messages.append({"role": "assistant", "name": st.session_state.bot_name, "content": reply})