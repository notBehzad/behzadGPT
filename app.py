import os
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import sentencepiece as sp
import streamlit as st

# =============================================================================
# 1. MODEL ARCHITECTURE (model_def.py)
# =============================================================================

CONFIG = {
    # Architecture
    "vocab_size": 8192,
    "d_model": 448,
    "d_ff": 1280,
    "context_length": 512,
    "d_head": 64,
    "layer_count": 9,
}


def precompute_rope_freqs(d_head, seq_len):
    i = torch.arange(0, d_head // 2, dtype=torch.float32)
    theta = 10000.0 ** (-2 * i / d_head)
    m = torch.arange(0, seq_len, dtype=torch.float32)

    freqs = torch.outer(m, theta)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return torch.sin(freqs), torch.cos(freqs)


def apply_RoPE(x, sin, cos):
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    half_d = x.shape[-1] // 2
    rotated_x = torch.cat([-x[..., half_d:], x[..., :half_d]], dim=-1)
    return (x * cos) + (rotated_x * sin)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, context_length):
        super().__init__()
        if d_model % d_head != 0:
            raise ValueError("d_model and d_head are not divisible.")
        self.num_heads = d_model // d_head
        self.d_head = d_head
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, sin, cos):
        B, S, _ = x.shape

        Q = apply_RoPE(self.Q(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2), sin[:S], cos[:S])
        K = apply_RoPE(self.K(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2), sin[:S], cos[:S])
        V = self.V(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        logits = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5)

        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(logits, dim=-1)
        heads_out = attn_weights @ V

        heads_out = heads_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.O(heads_out)


class SwiGELU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, seq_len, d_ff):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, d_head, seq_len)
        self.ffn = SwiGELU_FFN(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attention.O.special_init = True
        self.ffn.down.special_init = True

    def forward(self, x, sin, cos):
        x = x + self.attention(self.norm1(x), sin, cos)
        return x + self.ffn(self.norm2(x))


class behzadGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        sin, cos = precompute_rope_freqs(config["d_head"], config["context_length"])
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

        self.layers = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["d_head"],
                config["context_length"],
                config["d_ff"]
            ) for _ in range(config["layer_count"])
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, self.sin, self.cos)
        return torch.matmul(x, self.embeddings.weight.T)


# =============================================================================
# 2. GLOBAL CONSTANTS & DEVICE SETUP
# =============================================================================

MODEL_CHECKPOINT = os.environ.get("BEHZADGPT_CHECKPOINT", "model/model.pth")
TOKENIZER_PATH = os.environ.get("BEHZADGPT_TOKENIZER", "tokenizer/unigram_8k.model")
EOT_STRING = "<EOT>"
FALLBACK_EOT_ID = 3
MAX_NEW_TOKENS = 300
TOP_K = 4

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
amp_enabled = torch.cuda.is_available()
model_lock = threading.Lock()


# =============================================================================
# 3. MODEL & TOKENIZER LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading behzadGPT model & tokenizer...")
def load_model_and_tokenizer():
    if not os.path.exists(MODEL_CHECKPOINT):
        st.error(
            f"Checkpoint not found at `{MODEL_CHECKPOINT}`. "
            "Set `BEHZADGPT_CHECKPOINT` environment variable or place `model.pth` in `./model/`."
        )
        st.stop()
    if not os.path.exists(TOKENIZER_PATH):
        st.error(
            f"Tokenizer not found at `{TOKENIZER_PATH}`. "
            "Set `BEHZADGPT_TOKENIZER` environment variable or place `unigram_8k.model` in `./tokenizer/`."
        )
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


# =============================================================================
# 4. GENERATION INFERENCE LOGIC
# =============================================================================

@torch.no_grad()
def generate(model, tokenizer, eot_id, context_text: str, bot_name: str) -> tuple[str, str]:
    token_id = tokenizer.encode_as_ids(context_text[-512:])

    for _ in range(MAX_NEW_TOKENS):
        window = token_id[-CONFIG["context_length"]:]
        tokens = torch.tensor(window, dtype=torch.int64).unsqueeze(0).to(device)

        with autocast(device_type=device_type, enabled=amp_enabled):
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


# =============================================================================
# 5. STREAMLIT PAGE CONFIG & CUSTOM CSS (ORIGINAL UI STYLING)
# =============================================================================

st.set_page_config(
    page_title="behzadGPT",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #0F1418;
        --panel: #161D24;
        --panel-raised: #1C242C;
        --line: #2A333B;
        --paper: #E8E3D8;
        --paper-dim: #9AA4AC;
        --brass: #C9974A;
        --brass-dim: #8A6E3E;
        --signal: #7FA66B;
        --mono: 'JetBrains Mono', ui-monospace, monospace;
        --sans: 'Inter', -apple-system, sans-serif;
    }

    /* Base Page Overrides */
    .stApp {
        background-color: var(--ink) !important;
        color: var(--paper) !important;
        font-family: var(--sans) !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 800px !important;
    }

    /* Telemetry Bar */
    .telemetry {
        font-family: var(--mono);
        font-size: 11px;
        letter-spacing: 0.06em;
        color: var(--brass-dim);
        background: var(--panel);
        border-bottom: 1px solid var(--line);
        padding: 8px 20px;
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-bottom: 20px;
        margin-left: -1rem;
        margin-right: -1rem;
    }
    .telemetry b { color: var(--brass); font-weight: 500; }
    .telemetry .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--signal);
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 6px var(--signal);
    }

    /* Card Box */
    .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 28px;
        margin-top: 10px;
    }
    .card h1 {
        font-family: var(--mono);
        font-size: 20px;
        font-weight: 600;
        margin: 0 0 4px;
        color: var(--paper);
    }
    .card .sub {
        font-size: 18px;
        color: var(--paper-dim);
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Custom Labels & Inputs */
    label, .stTextArea label, .stTextInput label {
        font-family: var(--mono) !important;
        font-size: 11px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--brass) !important;
    }

    div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
        background-color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        color: var(--paper) !important;
    }

    input, textarea {
        color: var(--paper) !important;
        font-family: var(--sans) !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: var(--mono) !important;
        font-size: 13px !important;
        letter-spacing: 0.04em !important;
        background-color: var(--brass) !important;
        color: #1a1408 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        transition: background .15s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #dba85c !important;
    }

    /* Header Chat Info */
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .chat-header .who {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .avatar {
        width: 32px; height: 32px;
        border-radius: 6px;
        background: linear-gradient(135deg, var(--brass), var(--brass-dim));
        display: flex; align-items: center; justify-content: center;
        font-family: var(--mono);
        font-weight: 600;
        font-size: 14px;
        color: #1a1408;
    }
    .who-name { font-family: var(--mono); font-size: 14px; font-weight: 600; color: var(--paper); }
    .who-sub { font-family: var(--mono); font-size: 10.5px; color: var(--paper-dim); }

    /* Messages Display */
    .msg-wrap {
        display: flex;
        flex-direction: column;
        margin-bottom: 12px;
    }
    .msg {
        max-width: 80%;
        padding: 11px 14px;
        border-radius: 10px;
        font-size: 14.5px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .msg-label {
        font-family: var(--mono);
        font-size: 9.5px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.65;
        margin-bottom: 4px;
    }
    .msg.bot {
        align-self: flex-start;
        background: var(--panel-raised);
        border: 1px solid var(--line);
        border-bottom-left-radius: 2px;
        color: var(--paper);
    }
    .msg.user {
        align-self: flex-end;
        background: var(--brass);
        color: #1a1408;
        border-bottom-right-radius: 2px;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 6. APPLICATION STATE & MAIN INTERFACE
# =============================================================================

model, tokenizer, eot_id = load_model_and_tokenizer()

# Session State Initialization
if "session_active" not in st.session_state:
    st.session_state.session_active = False
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "context" not in st.session_state:
    st.session_state.context = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = "Behzad"
if "bot_name" not in st.session_state:
    st.session_state.bot_name = "Mario"

# Render Telemetry Strip
st.markdown(
    """
    <div class="telemetry">
        <span><span class="dot"></span><b>behzadGPT</b></span>
        <span>d_model <b>448</b></span>
        <span>layers <b>9</b></span>
        <span>heads <b>7</b></span>
        <span>vocab <b>8,192</b></span>
        <span>ctx <b>512</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SCREEN 1: SETUP SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.session_active:
    st.markdown(
        """
        <div class="card">
            <h1>New session</h1>
            <p class="sub">Set the scene before you start talking. This becomes the model's system prompt and won't change mid-conversation, start a new session to change it. Keep it simple and within 3 lines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_prompt = "Mario is a very good friend. He always helps Behzad and loves to talk to him. Today Mario met Behzad."

    with st.form("setup_form"):
        system_prompt = st.text_area(
            "Roleplay / system prompt",
            value=default_prompt,
            height=110,
        )

        col1, col2 = st.columns(2)
        with col1:
            user_name_input = st.text_input("Your name", value="Behzad")
        with col2:
            bot_name_input = st.text_input("Bot's name", value="Mario")

        start_submitted = st.form_submit_button("Start chat")

        if start_submitted:
            if not system_prompt.strip():
                st.error("Give the model a scenario to work with.")
            else:
                user_name = user_name_input.strip() or "You"
                bot_name = bot_name_input.strip() or "Bot"

                opening_context = f"System: {system_prompt.strip()} {EOT_STRING} {bot_name}: "

                with model_lock:
                    with st.spinner("Generating opening reply..."):
                        reply, decoded = generate(model, tokenizer, eot_id, opening_context, bot_name)

                st.session_state.user_name = user_name
                st.session_state.bot_name = bot_name
                st.session_state.context = decoded
                st.session_state.chat_history = []
                if reply:
                    st.session_state.chat_history.append({"role": "bot", "name": bot_name, "text": reply})
                st.session_state.session_active = True
                st.rerun()

# -----------------------------------------------------------------------------
# SCREEN 2: CHAT INTERFACE
# -----------------------------------------------------------------------------
else:
    col_hdr_left, col_hdr_right = st.columns([4, 1])

    with col_hdr_left:
        avatar_char = st.session_state.bot_name[0].upper() if st.session_state.bot_name else "B"
        st.markdown(
            f"""
            <div class="chat-header">
                <div class="who">
                    <div class="avatar">{avatar_char}</div>
                    <div>
                        <div class="who-name">{st.session_state.bot_name}</div>
                        <div class="who-sub">chatting as {st.session_state.user_name}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_hdr_right:
        if st.button("New session"):
            st.session_state.session_active = False
            st.session_state.chat_history = []
            st.session_state.context = ""
            st.rerun()

    # Render Chat History
    for msg in st.session_state.chat_history:
        role_class = "bot" if msg["role"] == "bot" else "user"
        st.markdown(
            f"""
            <div class="msg-wrap">
                <div class="msg {role_class}">
                    <div class="msg-label">{msg["name"]}</div>
                    {msg["text"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chat Input Box
    if user_message := st.chat_input("Type a message..."):
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "name": st.session_state.user_name, "text": user_message})

        prompt = f"{st.session_state.context} {st.session_state.user_name}: {user_message.strip()} {EOT_STRING} {st.session_state.bot_name}: "

        with model_lock:
            reply, decoded = generate(model, tokenizer, eot_id, prompt, st.session_state.bot_name)

        st.session_state.context = decoded
        st.session_state.chat_history.append({"role": "bot", "name": st.session_state.bot_name, "text": reply or "..."})
        st.rerun()