#!/usr/bin/env python3
"""Chunks raw text files into ~2000-character JSONL records."""
import sys, json

CHUNK_SIZE = 2000

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 chunk_corpus.py <input.txt> <output.jsonl>")
        sys.exit(1)
        
    with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    paragraphs = text.split("\n\n")
    current_chunk, current_len = [], 0

    with open(sys.argv[2], "w", encoding="utf-8") as out:
        for para in paragraphs:
            clean = " ".join(para.split())
            if not clean:
                continue
            if current_len + len(clean) > CHUNK_SIZE and current_chunk:
                out.write(json.dumps({"text": " ".join(current_chunk)}) + "\n")
                current_chunk, current_len = [], 0
            current_chunk.append(clean)
            current_len += len(clean)

        if current_chunk:
            out.write(json.dumps({"text": " ".join(current_chunk)}) + "\n")

if __name__ == "__main__":
    main()
