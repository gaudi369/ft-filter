use crate::model::Model;
use crate::tokenize::{fasttext_hash, get_subword_hashes, tokens};

pub struct InferSession {
    accumulator: Vec<f32>,
    word_hashes: Vec<i32>,
    logits: Vec<f32>,
}

impl InferSession {
    pub fn new(dim: usize) -> Self {
        Self {
            accumulator: vec![0.0f32; dim],
            word_hashes: Vec::with_capacity(256),
            logits: Vec::new(),
        }
    }

    #[inline(always)]
    fn add_row(acc: &mut [f32], row: &[f32]) {
        for (a, r) in acc.iter_mut().zip(row.iter()) {
            *a += *r;
        }
    }

    #[inline(always)]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn std_log(x: f32) -> f32 {
        // Upstream's 1e-5 literal promotes the addition and log to double.
        (f64::from(x) + 1e-5).ln() as f32
    }

    fn sigmoid(x: f32) -> f32 {
        (1.0f64 / f64::from(1.0f32 + (-x).exp())) as f32
    }

    fn table_sigmoid(x: f32) -> f32 {
        if x < -8.0 {
            0.0
        } else if x > 8.0 {
            1.0
        } else {
            let index = ((x + 8.0) * 32.0) as usize;
            let grid_x = index as f32 / 32.0 - 8.0;
            (1.0f64 / (1.0 + f64::from((-grid_x).exp()))) as f32
        }
    }

    pub fn predict_label_prob(&mut self, text: &str, model: &Model, label_idx: usize) -> f32 {
        self.accumulator.fill(0.0);
        self.word_hashes.clear();
        let dim = model.args.dim;
        let mut total_features = 0usize;
        let accumulator = &mut self.accumulator;
        let mut add_feature = |id: usize| {
            let offset = id * dim;
            Self::add_row(accumulator, &model.win[offset..offset + dim]);
            total_features += 1;
        };

        // Match Dictionary::getLine: word + its subwords in token order, then
        // word n-grams. Like Python predict, append EOS to each document.
        for token in tokens(text).chain(std::iter::once("</s>")) {
            let wid = model.word2id.get(token).copied();
            let is_label = model.label2id.contains_key(token)
                || (wid.is_none() && token.starts_with("__label__"));
            if !is_label {
                self.word_hashes
                    .push(fasttext_hash(token.as_bytes()) as i32);
                if let Some(id) = wid {
                    add_feature(id);
                }
                get_subword_hashes(
                    token,
                    model.args.minn,
                    model.args.maxn,
                    model.args.bucket,
                    |b| {
                        if let Some(id) = model.bucket_id(b) {
                            add_feature(id);
                        }
                    },
                );
            }
            if token == "</s>" {
                break;
            }
        }

        if model.args.word_ngrams > 1 && model.args.bucket > 0 {
            for i in 0..self.word_hashes.len() {
                // C++ converts signed int32 hashes to uint64 (sign extension).
                let mut h = self.word_hashes[i] as u64;
                let end = self
                    .word_hashes
                    .len()
                    .min(i + model.args.word_ngrams as usize);
                for j in i + 1..end {
                    h = h
                        .wrapping_mul(116049371)
                        .wrapping_add(self.word_hashes[j] as u64);
                    if let Some(id) = model.bucket_id((h % model.args.bucket as u64) as usize) {
                        add_feature(id);
                    }
                }
            }
        }
        if total_features == 0 {
            return 0.0;
        }
        let scale = 1.0 / total_features as f32;
        for v in &mut self.accumulator {
            *v *= scale;
        }

        if model.args.loss == 1 {
            // Reference prediction accumulates smoothed log probabilities from
            // root to leaf, rather than multiplying probabilities leaf-first.
            let mut score = 0.0f32;
            for (&node, &right) in model.paths[label_idx]
                .iter()
                .zip(&model.codes[label_idx])
                .rev()
            {
                let off = node * dim;
                let logit = Self::dot_product(&self.accumulator, &model.wout[off..off + dim]);
                let s = Self::sigmoid(logit);
                score += Self::std_log(if right { s } else { 1.0 - s });
                // Upstream DFS prunes even at threshold=0 using std_log(0).
                if score < Self::std_log(0.0) {
                    return 0.0;
                }
            }
            score.exp()
        } else if model.args.loss == 3 {
            self.logits.resize(model.num_labels, 0.0);
            let mut max_score = f32::NEG_INFINITY;
            for (l, logit) in self.logits.iter_mut().enumerate() {
                let off = l * dim;
                *logit = Self::dot_product(&self.accumulator, &model.wout[off..off + dim]);
                max_score = max_score.max(*logit);
            }
            let mut sum_exp = 0.0f32;
            for logit in &mut self.logits {
                *logit = f64::from(*logit - max_score).exp() as f32;
                sum_exp += *logit;
            }
            Self::std_log(self.logits[label_idx] / sum_exp).exp()
        } else {
            let off = label_idx * dim;
            let logit = Self::dot_product(&self.accumulator, &model.wout[off..off + dim]);
            Self::std_log(Self::table_sigmoid(logit)).exp()
        }
    }
}
