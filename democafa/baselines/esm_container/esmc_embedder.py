#!/usr/bin/env python3
"""Generate per-protein ESM C embeddings (mean-pooled) and save them to HDF5.

Output layout matches prott5_embedder.py: one dataset per protein,
key = UniProt accession, value = 1-D float32 vector.

ProtT5 (--model_name prot_t5) is also supported so the two language models can
be compared through the exact same downstream pipeline.
"""
import argparse
import gzip
import time

import h5py
import numpy as np
import torch
from esm.models.esmc import ESMC
from tqdm import tqdm

PROT_T5 = "Rostlab/prot_t5_xl_half_uniref50-enc"
MODEL_CHOICES = ["esmc_300m", "esmc_600m", "prot_t5"]


def parse_id(header):
    """'>sp|Q7L5A3|ATOSB_HUMAN ...' -> 'Q7L5A3'; '>Q7L5A3 ...' -> 'Q7L5A3'."""
    first = header[1:].split()[0]
    parts = first.split("|")
    if len(parts) >= 3 and parts[0] in ("sp", "tr"):
        return parts[1]
    return first


def read_fasta(path):
    """Read a plain or gzipped FASTA into {accession: sequence}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs, pid = {}, None
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                pid = parse_id(line)
                seqs[pid] = []
            else:
                # drop alignment gaps and stop codons
                seqs[pid].append(line.upper().replace("-", "").replace("*", ""))
    return {k: "".join(v) for k, v in seqs.items()}


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batches(items, max_residues, max_batch):
    """Group (id, seq) pairs so each batch stays under max_residues / max_batch."""
    batch, n_res = [], 0
    for pid, seq in items:
        if batch and (n_res + len(seq) > max_residues or len(batch) >= max_batch):
            yield batch
            batch, n_res = [], 0
        batch.append((pid, seq))
        n_res += len(seq)
    if batch:
        yield batch


@torch.no_grad()
def embed_batch(model, batch, layers=()):
    """Return {id: {view: mean-pooled embedding}} for one batch of (id, seq).

    view "last" is the final (normalized) output; each n in layers adds the
    output of transformer block n (1-based, before the final norm).
    """
    seqs = [s for _, s in batch]
    tokens = model._tokenize(seqs)              # (B, L_max + 2): <cls> seq <eos> <pad>...
    out = model(sequence_tokens=tokens)         # padding is masked inside forward()
    views = {"last": out.embeddings}            # each (B, L_max + 2, D)
    for n in layers:
        views[n] = out.hidden_states[n - 1]

    result = {}
    for i, (pid, seq) in enumerate(batch):
        # drop <cls>, <eos> and padding; the model runs in bfloat16 on GPU/MPS
        # and numpy has no bf16, so cast first
        result[pid] = {v: e[i, 1:len(seq) + 1].float().mean(dim=0).cpu().numpy()
                       for v, e in views.items()}
    return result


class ProtT5Embedder:
    """ProtT5 encoder wrapped to look like the ESM C model for embed_batch_t5."""

    def __init__(self, device):
        from transformers import T5EncoderModel, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(PROT_T5, do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(PROT_T5)
        # the checkpoint is fp16; T5 can overflow in fp16 off CUDA, so use fp32 there
        model = model.to(torch.float16 if device.type == "cuda" else torch.float32)
        self.model = model.to(device).eval()
        self.device = device


@torch.no_grad()
def embed_batch_t5(t5, batch, layers=()):
    """Same contract as embed_batch (last layer only), for ProtT5 (rare residues mapped to X, as in prott5_embedder.py)."""
    seqs = [" ".join(s.replace("U", "X").replace("Z", "X").replace("O", "X")) for _, s in batch]
    enc = t5.tokenizer(seqs, add_special_tokens=True, padding="longest", return_tensors="pt")
    out = t5.model(input_ids=enc["input_ids"].to(t5.device),
                   attention_mask=enc["attention_mask"].to(t5.device))
    emb = out.last_hidden_state                 # (B, L_max + 1, 1024): seq </s> <pad>...

    result = {}
    for i, (pid, seq) in enumerate(batch):
        per_res = emb[i, :len(seq)]             # drop </s> and padding
        result[pid] = {"last": per_res.float().mean(dim=0).cpu().numpy()}
    return result


def layer_path(out_path, layer):
    """'x.h5' -> 'x.L15.h5' for layer 15; the last layer keeps out_path."""
    return out_path if layer == "last" else f"{str(out_path)[:-3]}.L{layer}.h5"


def get_embeddings(fasta, out_path, model_name="esmc_300m", device="auto",
                   max_len=2048, max_residues=4000, max_batch=100, layers=()):
    device = get_device(device)
    print(f"Using device: {device} | model: {model_name}")
    if model_name == "prot_t5":
        if layers:
            raise ValueError("--layers is only supported for ESM C models")
        model, embed = ProtT5Embedder(device), embed_batch_t5
    else:
        model, embed = ESMC.from_pretrained(model_name, device=device).eval(), embed_batch
        n_blocks = len(model.transformer.blocks)
        bad = [n for n in layers if not 1 <= n <= n_blocks]
        if bad:
            raise ValueError(f"layers {bad} out of range: {model_name} has {n_blocks} blocks")
    if layers and not str(out_path).endswith(".h5"):
        raise ValueError("output must end with .h5 when --layers is used")
    views = ["last", *layers]

    seqs = read_fasta(fasta)
    items, n_trunc = [], 0
    for pid, seq in seqs.items():
        if len(seq) > max_len:                  # attention is O(L^2) in memory
            seq, n_trunc = seq[:max_len], n_trunc + 1
        items.append((pid, seq))
    # longest first, so sequences of similar length share a batch (less padding)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    print(f"Read {len(items)} sequences from {fasta}; {n_trunc} truncated to {max_len} residues")

    start, n_done = time.time(), 0
    # one HDF5 per view, written as we go to keep memory flat
    files = {v: h5py.File(layer_path(out_path, v), "w") for v in views}
    try:
        for batch in tqdm(list(make_batches(items, max_residues, max_batch)), desc="Embedding"):
            try:
                embs = embed(model, batch, layers)
            except torch.OutOfMemoryError:
                # fall back to one sequence at a time for this batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                embs = {}
                for one in batch:
                    try:
                        embs.update(embed(model, [one], layers))
                    except torch.OutOfMemoryError:
                        print(f"Skipped {one[0]} (L={len(one[1])}): out of memory")
            for pid, by_view in embs.items():
                for v, vec in by_view.items():
                    files[v].create_dataset(pid, data=vec.astype(np.float32))
            n_done += len(embs)
    finally:
        for hf in files.values():
            hf.close()

    elapsed = time.time() - start
    outputs = ", ".join(layer_path(out_path, v) for v in views)
    print(f"Embedded {n_done}/{len(items)} proteins in {elapsed:.1f}s -> {outputs}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate mean-pooled ESM C embeddings for a FASTA file.")
    p.add_argument("-i", "--input", required=True, help="Input FASTA (plain or .gz)")
    p.add_argument("-o", "--output", required=True, help="Output HDF5 file")
    p.add_argument("--model_name", default="esmc_300m", choices=MODEL_CHOICES)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (default: auto)")
    p.add_argument("--max_len", type=int, default=2048, help="Truncate sequences longer than this")
    p.add_argument("--max_residues", type=int, default=4000, help="Max total residues per batch")
    p.add_argument("--max_batch", type=int, default=100, help="Max sequences per batch")
    p.add_argument("--layers", default="",
                   help="Also save these transformer blocks (1-based, comma-separated, ESM C only), "
                        "e.g. 10,15,20 -> <output>.L10.h5 ...; the last layer always goes to --output")
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    get_embeddings(args.input, args.output, args.model_name, args.device,
                   args.max_len, args.max_residues, args.max_batch, layers)


if __name__ == "__main__":
    main()
