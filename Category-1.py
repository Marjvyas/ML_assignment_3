import streamlit as st
import torch
import torch.nn as nn
import json
import io
import tempfile
import re
from typing import Tuple

# ---------------------------
# Sidebar: select model params
# ---------------------------
st.sidebar.title("Model Selection")

st.sidebar.write("Model-1: Context=3, Embedding=64, Activation=ReLU")
st.sidebar.write("Model-2: Context=5, Embedding=32, Activation=ReLU")

CONTEXT_SIZE = st.sidebar.radio("Context Size", [3, 5], index=0, horizontal=True)
EMBEDDING_DIM = st.sidebar.radio("Embedding Dim", [32, 64], index=1, horizontal=True)
ACTIVATION_FUNCTION = st.sidebar.radio("Activation Function", ["ReLU", "Tanh"], index=0, horizontal=True)
LOAD_BUTTON = st.sidebar.button("Load Model")

# Map parameter combos -> filenames (update names to your actual files)
MODEL_FILENAME_MAP = {
    (3, 64, "ReLU"): "best_model.pth",
    (5, 32, "ReLU"): "best_model (1).pth",
}

HIDDEN_DIM = 512  # must match training


# ---------------------------
# MLP architecture (must match training)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int, hidden_dim: int, activation_fn: nn.Module):
        """
        NOTE: constructor order: (vocab_size, embedding_dim, context_length, hidden_dim, activation_fn)
        activation_fn should be an nn.Module instance (e.g., nn.ReLU()).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = embedding_dim * context_length
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, vocab_size)
        self.act = activation_fn
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        # x: (batch, context_length)
        emb = self.embedding(x)                 # (batch, context_length, embedding_dim)
        emb_flat = emb.view(emb.size(0), -1)    # (batch, embedding_dim * context_length)
        h1 = self.act(self.bn1(self.l1(emb_flat)))
        h2 = self.act(self.bn2(self.l2(h1)))
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)
        h2 = h2 + h1
        logits = self.l3(h2)                    # (batch, vocab_size)
        return logits


# ---------------------------
# JSON loading and normalize
# ---------------------------
@st.cache_data
def load_json_from_bytes(json_bytes: bytes):
    try:
        return json.loads(json_bytes.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to decode JSON: {e}")


def normalize_stoi_itos(stoi_obj, itos_obj):
    """
    Return (stoi: dict[str->int], itos: dict[str->str])
    Accepts itos as list or dict.
    """
    if not isinstance(stoi_obj, dict):
        raise RuntimeError("stoi.json must be an object mapping token->index.")
    # keep keys as strings
    stoi = {str(k): int(v) for k, v in stoi_obj.items()}

    if isinstance(itos_obj, list):
        itos = {str(i): tok for i, tok in enumerate(itos_obj)}
    elif isinstance(itos_obj, dict):
        itos = {str(int(k)): v for k, v in itos_obj.items()}
    else:
        raise RuntimeError("itos.json must be a list or dict.")
    return stoi, itos


# ---------------------------
# Model loader (cached) - pass activation_name (string)
# ---------------------------
@st.cache_resource
def load_model_from_bytes(
    model_bytes: bytes,
    vocab_size: int,
    emb_dim: int,
    context_size: int,
    hidden_dim: int,
    activation_name: str = "relu",
) -> Tuple[torch.nn.Module, str]:
    """
    Load model from bytes. activation_name is a string ('relu' or 'tanh') to avoid unhashable args.
    Returns (model, load_type_str).
    """
    # create activation module from name (hashable string passed in)
    act = None
    if activation_name.lower() == "relu":
        act = nn.ReLU()
    elif activation_name.lower() == "tanh":
        act = nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")

    # Try TorchScript first (write temp file)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(model_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()
        try:
            model_ts = torch.jit.load(tmp_path, map_location="cpu")
            model_ts.eval()
            return model_ts, "torchscript"
        except Exception:
            pass
    finally:
        try:
            tmp.close()
        except Exception:
            pass

    # Try torch.load from bytes
    buf = io.BytesIO(model_bytes)
    try:
        loaded = torch.load(buf, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"torch.load failed: {e}")

    if isinstance(loaded, torch.nn.Module):
        loaded.eval()
        return loaded, "module_object"

    # If dict: find state_dict or nested
    if isinstance(loaded, dict):
        state = None
        if "model_state_dict" in loaded:
            state = loaded["model_state_dict"]
        elif "state_dict" in loaded:
            state = loaded["state_dict"]
        else:
            # maybe raw state_dict (tensor values)
            if all(isinstance(v, torch.Tensor) for v in loaded.values()):
                state = loaded
            else:
                # try to find nested dict with tensors
                for k, v in loaded.items():
                    if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                        state = v
                        break

        if state is None:
            raise RuntimeError(f"No state_dict found in checkpoint. Keys: {list(loaded.keys())}")

        # instantiate model with activation module
        model = MLP(vocab_size, emb_dim, context_size, hidden_dim, act)
        try:
            model.load_state_dict(state)
            model.eval()
            return model, "state_dict_loaded"
        except Exception as e:
            # helpful diagnostic
            ck_shapes = {k: tuple(v.shape) for k, v in state.items()}
            model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
            raise RuntimeError(
                f"Error loading state_dict: {e}\n\n"
                "Checkpoint sample shapes:\n"
                + "\n".join([f"  {k}: {ck_shapes[k]}" for k in list(ck_shapes)[:8]])
                + "\n\nModel expected shapes:\n"
                + "\n".join([f"  {k}: {model_shapes[k]}" for k in list(model_shapes)[:8]])
                + "\n\nMake sure model class matches training architecture and constructor order."
            )

    raise RuntimeError(f"Unsupported object loaded by torch.load: {type(loaded)}")


# ---------------------------
# Generation function (supports temperature sampling)
# ---------------------------
def generate_text(
    model: torch.nn.Module,
    stoi: dict,
    itos: dict,
    prompt: str,
    max_new_words: int,
    context_size: int,
    device: torch.device,
    mode: str = "Greedy",
    temperature: float = 1.0,
) -> str:
    """
    mode: "Greedy" or "Sampling"
    temperature: used only when mode == "Sampling" (must be > 0)
    """
    # robust token index lookup (case-insensitive + special tokens)
    def get_token_idx(token: str, default: int):
        if token in stoi:
            return stoi[token]
        if token.lower() in stoi:
            return stoi[token.lower()]
        if token.upper() in stoi:
            return stoi[token.upper()]
        return default

    pad_token_idx = get_token_idx("<PAD>", 0)
    unk_token_idx = get_token_idx("<UNK>", 1)

    # Basic tokenization preserving tokens like <BOS>, <EOS>
    raw_tokens = re.findall(r"<[^>]+>|[a-zA-Z0-9']+", prompt)
    tokens = [t.lower() for t in raw_tokens]

    # Convert to indices with OOV -> <UNK>
    indices = [stoi.get(t, unk_token_idx) for t in tokens]

    if any(t not in stoi for t in tokens):
        oov = [t for t in tokens if t not in stoi]
        st.warning(f"OOV tokens replaced with <UNK>: {', '.join(sorted(set(oov))) }")

    generated = list(indices)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_words):
            context_indices = generated[-context_size:]
            if len(context_indices) < context_size:
                padded = [pad_token_idx] * (context_size - len(context_indices)) + context_indices
            else:
                padded = context_indices

            input_tensor = torch.tensor([padded], dtype=torch.long).to(device)  # (1, context_size)
            out = model(input_tensor)

            # resolve logits
            if isinstance(out, (list, tuple)):
                logits = out[0]
            elif isinstance(out, dict):
                logits = out.get("logits") or next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
                if logits is None:
                    raise RuntimeError("Model returned dict with no tensor logits.")
            elif isinstance(out, torch.Tensor):
                logits = out
            else:
                raise RuntimeError("Unrecognized model output type.")

            if logits.dim() == 2:
                last_logits = logits  # (batch, vocab)
            elif logits.dim() == 3:
                last_logits = logits[:, -1, :]  # (batch, vocab)
            else:
                raise RuntimeError(f"Unexpected logits dim: {logits.dim()}")

            # choose next index based on mode
            if mode == "Greedy":
                # deterministic
                next_idx = int(torch.argmax(last_logits, dim=-1).squeeze().cpu().item())
            else:
                # Sampling mode: apply temperature, then sample
                if temperature <= 0:
                    raise ValueError("Temperature must be > 0 for sampling.")
                scaled = last_logits / float(temperature)
                probs = torch.softmax(scaled, dim=-1)  # (batch, vocab)
                # sample one token from the distribution
                next_idx = int(torch.multinomial(probs.squeeze(0), num_samples=1).squeeze().cpu().item())

            generated.append(next_idx)

    # convert indices back to tokens using itos (string keys)
    tok_list = [itos.get(str(i), "<UNK>") for i in generated]
    return " ".join(tok_list)


# ---------------------------
# App main (session-state based load)
# ---------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Next Word Prediction — Category-1")

    # Initialize session_state keys if missing
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "model_bytes" not in st.session_state:
        st.session_state.model_bytes = None
    if "model_path" not in st.session_state:
        st.session_state.model_path = None
    if "stoi" not in st.session_state:
        st.session_state.stoi = None
    if "itos" not in st.session_state:
        st.session_state.itos = None
    if "load_type" not in st.session_state:
        st.session_state.load_type = None
    if "config" not in st.session_state:
        st.session_state.config = None

    # Handle Load Model button press
    if LOAD_BUTTON:
        config_key = (CONTEXT_SIZE, EMBEDDING_DIM, ACTIVATION_FUNCTION)
        model_path = MODEL_FILENAME_MAP.get(config_key)
        if model_path is None:
            st.error("No model file mapped for this configuration. Update MODEL_FILENAME_MAP.")
        else:
            try:
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
            except FileNotFoundError:
                st.error(f"Model file not found: {model_path}")
                model_bytes = None

            # load stoi/itos from local files
            try:
                with open("stoi.json", "rb") as f:
                    stoi_bytes = f.read()
                with open("itos.json", "rb") as f:
                    itos_bytes = f.read()
            except FileNotFoundError as e:
                st.error(f"Missing file: {e.filename}. Provide stoi.json and itos.json next to script.")
                stoi_bytes = None
                itos_bytes = None

            if model_bytes and stoi_bytes and itos_bytes:
                try:
                    stoi_obj = load_json_from_bytes(stoi_bytes)
                    itos_obj = load_json_from_bytes(itos_bytes)
                    stoi, itos = normalize_stoi_itos(stoi_obj, itos_obj)
                except Exception as e:
                    st.error(f"Failed loading stoi/itos: {e}")
                    stoi = None
                    itos = None

                if stoi and itos:
                    # Save to session state (model bytes + vocab); actual model instance will be created via cached loader later
                    st.session_state.model_bytes = model_bytes
                    st.session_state.model_path = model_path
                    st.session_state.stoi = stoi
                    st.session_state.itos = itos
                    st.session_state.config = {"context": CONTEXT_SIZE, "emb": EMBEDDING_DIM, "act": ACTIVATION_FUNCTION}
                    st.session_state.model_loaded = True
                    st.success(f"Files loaded into session for {model_path}. Now click Generate to produce text.")
                    

    # If already loaded (from earlier click), show info
    if st.session_state.model_loaded:
        st.sidebar.success(f"Loaded: {st.session_state.model_path}")
        st.sidebar.write(f"Config: {st.session_state.config}")

    # generation UI
   
    prompt = st.text_area("Enter your prompt:", "", height=120)

    # Mode selection: Greedy vs Sampling
    gen_mode = st.radio("Generation mode:", ["Greedy", "Sampling"], index=0, horizontal=True)
    temperature = 1.0
    if gen_mode == "Sampling":
        temperature = st.slider("Temperature (higher = more random)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)

    num_words = 1

    if st.button("Generate"):
        if not st.session_state.model_loaded:
            st.warning("Please select model parameters in the sidebar and click 'Load Model' first.")
            st.stop()

        if not prompt.strip():
            st.warning("Please enter a prompt.")
            st.stop()

        # Build activation name string and call cached loader (which returns model object)
        act_name = "relu" if st.session_state.config["act"].lower().startswith("relu") else "tanh"
        try:
            model, load_type = load_model_from_bytes(
                st.session_state.model_bytes,
                len(st.session_state.stoi),
                st.session_state.config["emb"],
                st.session_state.config["context"],
                HIDDEN_DIM,
                act_name,
            )
        except Exception as e:
            st.error("Failed to instantiate model from bytes:")
            st.text(str(e))
            st.stop()

        # move model to device where generation will happen
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.to(device)
        except Exception:
            pass

        try:
            out_text = generate_text(
                model,
                st.session_state.stoi,
                st.session_state.itos,
                prompt,
                num_words,
                st.session_state.config["context"],
                device,
                mode=gen_mode,
                temperature=temperature,
            )
            st.subheader("Generated Text")
            st.markdown(out_text)
        except Exception as e:
            st.error(f"Generation failed: {e}")

if __name__ == "__main__":
    main()
