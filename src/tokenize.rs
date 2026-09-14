#[inline(always)]
pub fn fasttext_hash(bytes: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in bytes {
        h = (h ^ (b as u32)).wrapping_mul(16777619);
    }
    h
}

/// Computes subword hashes strictly on UTF-8 byte slices, matching official FastText C++.
/// Character boundary n-grams are extracted over byte lengths between minn and maxn.
pub fn get_subword_hashes(
    word: &str,
    minn: usize,
    maxn: usize,
    bucket: usize,
    out_ids: &mut Vec<usize>,
) {
    if minn == 0 || maxn == 0 || minn > maxn || bucket == 0 {
        return;
    }

    let mut wrapped = Vec::with_capacity(word.len() + 2);
    wrapped.push(b'<');
    wrapped.extend_from_slice(word.as_bytes());
    wrapped.push(b'>');

    let total_bytes = wrapped.len();

    // Collect all char byte boundaries so n-grams don't split valid multi-byte UTF-8 codepoints
    let mut char_boundaries = Vec::with_capacity(total_bytes + 1);
    let mut idx = 0;
    while idx < total_bytes {
        char_boundaries.push(idx);
        let b = wrapped[idx];
        if b < 0x80 {
            idx += 1;
        } else if (b & 0xE0) == 0xC0 {
            idx += 2;
        } else if (b & 0xF0) == 0xE0 {
            idx += 3;
        } else {
            idx += 4;
        }
    }
    char_boundaries.push(total_bytes);

    let num_chars = char_boundaries.len() - 1;

    for i in 0..num_chars {
        for j in i..num_chars {
            let char_len = j - i + 1;
            if char_len >= minn && char_len <= maxn {
                // Official FastText rule: do not re-add the full original token boundary <word>
                if !(i == 0 && j == num_chars - 1) {
                    let start_byte = char_boundaries[i];
                    let end_byte = char_boundaries[j + 1];
                    let h = fasttext_hash(&wrapped[start_byte..end_byte]);
                    out_ids.push((h as usize) % bucket);
                }
            }
            if char_len > maxn {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fasttext_hash_parity() {
        assert_eq!(fasttext_hash(b"hello"), 1335831723);
        assert_eq!(fasttext_hash(b"<the>"), 2180892062);
    }
}
