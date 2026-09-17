#[inline(always)]
pub fn fasttext_hash(bytes: &[u8]) -> u32 {
    bytes.iter().fold(2166136261, |h, &b| hash_byte(h, b))
}

#[inline(always)]
fn hash_byte(h: u32, b: u8) -> u32 {
    // Upstream deliberately sign-extends bytes for compatibility with old models.
    (h ^ (b as i8 as u32)).wrapping_mul(16777619)
}

/// FastText splits only these ASCII bytes, not punctuation or Unicode whitespace.
/// JSONL documents are treated as a single line (embedded LF becomes whitespace).
pub fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split([' ', '\n', '\r', '\t', '\u{000b}', '\u{000c}', '\0'])
        .filter(|token| !token.is_empty())
}

/// Visit UTF-8 character n-grams of the virtual string `<word>` without allocating.
/// Only the standalone boundary characters are excluded, not the whole `<word>`.
pub fn get_subword_hashes(
    word: &str,
    minn: usize,
    maxn: usize,
    bucket: usize,
    mut emit: impl FnMut(usize),
) {
    if maxn == 0 || minn > maxn || bucket == 0 || word == "</s>" {
        return;
    }
    let bytes = word.as_bytes();
    let len = bytes.len() + 2;
    let byte_at = |i: usize| match i {
        0 => b'<',
        i if i == len - 1 => b'>',
        i => bytes[i - 1],
    };
    for i in 0..len {
        if byte_at(i) & 0xc0 == 0x80 {
            continue;
        }
        let mut j = i;
        let mut h = 2166136261;
        for n in 1..=maxn {
            if j == len {
                break;
            }
            h = hash_byte(h, byte_at(j));
            j += 1;
            while j < len && byte_at(j) & 0xc0 == 0x80 {
                h = hash_byte(h, byte_at(j));
                j += 1;
            }
            if n >= minn && !(n == 1 && (i == 0 || j == len)) {
                emit(h as usize % bucket);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_reference_vectors() {
        assert_eq!(fasttext_hash(b"hello"), 1335831723);
        assert_eq!(fasttext_hash(b"<the>"), 4243648960);
        assert_eq!(fasttext_hash(b"\xff"), 4193493326);
    }

    #[test]
    fn subwords_include_whole_word_but_not_single_boundaries() {
        let mut ids = Vec::new();
        get_subword_hashes("é", 0, 3, 2_000_000, |id| ids.push(id));
        let expected: Vec<_> = ["<é", "<é>", "é", "é>"]
            .iter()
            .map(|s| fasttext_hash(s.as_bytes()) as usize % 2_000_000)
            .collect();
        assert_eq!(ids, expected);
        ids.clear();
        get_subword_hashes("</s>", 1, 6, 2_000_000, |id| ids.push(id));
        assert!(ids.is_empty());
    }

    #[test]
    fn ascii_separators_only() {
        assert_eq!(
            tokens("a\0b\tc\nd\re\u{000b}f\u{000c}g a\u{00a0}b hi,there").collect::<Vec<_>>(),
            ["a", "b", "c", "d", "e", "f", "g", "a\u{00a0}b", "hi,there"]
        );
    }
}
