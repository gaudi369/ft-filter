#!/usr/bin/env python3
"""Offline parity regressions using tiny reference-trained FP32 models and lid.176.bin."""
import json
from pathlib import Path
import subprocess
import tempfile

import fasttext

BINARY = Path("target/release/ft-filter")
EPSILON = 1e-5
TEXTS = [
    "", " ", "hello", "the", "hello world", "hello,world!",
    "Bonjour le monde, où êtes-vous ?", "Hola señor, ¿cómo está usted?",
    "Привет мир", "你好世界", "日本語の文章です", "مرحبا بالعالم",
    "हिन्दी भाषा", "é", "🙂🚀", "café naïve über Straße",
    "hello\u00a0world", "hello\u2003world", "hello\0world",
    "hello\tworld\rbonjour\vmonde\f!", "hello\nworld", "hello\n\nworld\n",
    "__label__a hello __label__unknown world", "hello </s> ignored words",
    "outofvocabularyword", "hello world\u00a0",
    "hello world " * 2000,
]


def verify(path, texts, work, labels=None):
    model = fasttext.load_model(str(path))
    if labels is None:
        labels = model.labels
    reference = [dict(zip(*model.predict(text.replace("\n", " "), k=-1)))
                 for text in texts]
    source = work / "input.jsonl"
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    max_diff = 0.0
    for label in labels:
        result = subprocess.run([
            str(BINARY), "filter", "-m", str(path), "-i", str(source),
            "-l", label, "-s", "-t", "0",
        ], check=True, capture_output=True, text=True)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(records) == len(texts), (path, label, "missing records")
        for index, (record, text, scores) in enumerate(zip(records, texts, reference)):
            assert record["text"] == text
            actual = record["ft_score"]
            expected = float(scores.get(label, 0.0))
            diff = abs(actual - expected)
            max_diff = max(max_diff, diff)
            assert diff <= EPSILON, (path, label, index, text[:100], actual, expected, diff)
    print(f"[PASS] {path.name}: {len(texts)} texts x {len(labels)} labels; max diff {max_diff:.2e}")


def main():
    with tempfile.TemporaryDirectory(prefix="ft-filter-parity-") as tmp:
        work = Path(tmp)
        training = work / "train.txt"
        training.write_text((
            "__label__a hello world café é 🙂\n"
            "__label__b bonjour le monde naïve Straße\n"
            "__label__c 你好世界 Привет мир 日本語\n"
        ) * 30)
        # All heads; minn=0 is valid and must still emit character n-grams.
        for loss, minn, maxn, ngrams in [
            ("softmax", 0, 0, 3), ("softmax", 0, 4, 2),
            ("hs", 1, 4, 3), ("ova", 2, 5, 2), ("ns", 2, 4, 2),
        ]:
            path = work / f"{loss}-{minn}-{maxn}.bin"
            model = fasttext.train_supervised(
                input=str(training), dim=16, epoch=10, minCount=1,
                bucket=1000, minn=minn, maxn=maxn, wordNgrams=ngrams,
                loss=loss, thread=1, verbose=0,
            )
            model.save_model(str(path))
            verify(path, TEXTS, work)

        lid = Path("data/lid.176.bin")
        if not lid.exists():
            raise SystemExit("Missing data/lid.176.bin; supply the unquantized language-ID model.")
        # More than 100 multilingual samples, including top-ranked and low-score labels.
        texts = TEXTS + [f"{a} {b}" for a in TEXTS[2:12] for b in TEXTS[2:12]]
        verify(lid, texts, work, ["__label__" + lang for lang in
                                ["en", "fr", "es", "ru", "zh", "ja", "ar", "hi", "de"]])


if __name__ == "__main__":
    main()
