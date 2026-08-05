import os
import pandas as pd
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

PARQUET_PATH = 'valid.parquet'
TOKENIZER_MODEL = "../tokenizer/unigram_8k.model"

# Output binary files
INPUT_IDS_BIN = 'x_valid_fine_tune.bin'
LABELS_BIN = 'y_valid_fine_tune.bin'


def process_dataset():
    if not os.path.exists(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer model '{TOKENIZER_MODEL}' not found!")
    
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    print(f"Loaded tokenizer from {TOKENIZER_MODEL} (Vocab size: {sp.get_piece_size()})")

    print(f"Loading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)

    total_tokens = 0

    with open(INPUT_IDS_BIN, "wb") as f_ids, open(LABELS_BIN, "wb") as f_lbl:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing & Generating Labels"):
            narrative_raw = row.get('narrative', '')
            narrative = str(narrative_raw).strip() if pd.notna(narrative_raw) else ''

            dialogue = list(row.get('dialogue', [])) if row.get('dialogue') is not None else []
            speakers = list(row.get('speakers', [])) if row.get('speakers') is not None else []

            if not narrative or len(dialogue) == 0:
                continue

            target_speaker = narrative.split()[0].rstrip(".,!?:;'\"")

            sample_ids = []
            sample_labels = []

            sys_text = f"System: {narrative} <EOT>\n"
            sys_tokens = sp.encode(sys_text, out_type=int, add_bos=False, add_eos=False)

            sample_ids.extend(sys_tokens)
            sample_labels.extend([-100] * len(sys_tokens))

            for speaker, text in zip(speakers, dialogue):
                speaker_str = str(speaker).strip()
                text_str = str(text).strip()

                header_text = f"{speaker_str}: "
                header_tokens = sp.encode(header_text, out_type=int, add_bos=False, add_eos=False)

                utterance_text = f"{text_str} <EOT>\n"
                utterance_tokens = sp.encode(utterance_text, out_type=int, add_bos=False, add_eos=False)

                sample_ids.extend(header_tokens)
                sample_labels.extend([-100] * len(header_tokens))

                sample_ids.extend(utterance_tokens)
                if speaker_str.lower() == target_speaker.lower():
                    sample_labels.extend(utterance_tokens)
                else:
                    sample_labels.extend([-100] * len(utterance_tokens))

            sep_text = "<EOS>\n\n"
            sep_tokens = sp.encode(sep_text, out_type=int, add_bos=False, add_eos=False)

            sample_ids.extend(sep_tokens)
            sample_labels.extend([-100] * len(sep_tokens))

            assert len(sample_ids) == len(sample_labels), \
                f"CRITICAL ERROR: Array length mismatch at row {idx}!"

            total_tokens += len(sample_ids)

            np.array(sample_ids, dtype=np.int16).tofile(f_ids)
            np.array(sample_labels, dtype=np.int16).tofile(f_lbl)

    print(f"\nProcessing Complete!")
    print(f"Total Tokens Processed: {total_tokens:,}")
    print(f"Saved Files:\n - {INPUT_IDS_BIN} (int32)\n - {LABELS_BIN} (int32)")


def verify_alignment():
    """ Sanity check script to visually inspect token masking """
    print("\n" + "="*65)
    print("RUNNING SANITY VERIFICATION ON FIRST 100 TOKENS...")
    print("="*65)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    ids = np.fromfile(INPUT_IDS_BIN, dtype=np.int16, count=100)
    labels = np.fromfile(LABELS_BIN, dtype=np.int16, count=100)

    print(f"{'INDEX':<6} | {'TOKEN':<22} | {'LABEL ID':<10} | {'TRAINING STATUS'}")
    print("-" * 65)

    for i, (t_id, lbl) in enumerate(zip(ids, labels)):
        token_str = repr(sp.decode([int(t_id)]))
        status = "COMPUTE LOSS" if lbl != -100 else "IGNORED (-100)"
        print(f"{i:<6} | {token_str:<22} | {lbl:<10} | {status}")


if __name__ == "__main__":
    process_dataset()
    verify_alignment()