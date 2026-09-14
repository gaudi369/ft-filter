use crate::model::Model;
use crate::tokenize::get_subword_hashes;

pub struct InferSession {
    accumulator: Vec<f32>,
    word_ids: Vec<usize>,
    subword_buckets: Vec<usize>,
}

impl InferSession {
    pub fn new(dim: usize) -> Self {
        Self {
            accumulator: vec![0.0f32; dim],
            word_ids: Vec::with_capacity(256),
            subword_buckets: Vec::with_capacity(1024),
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

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn predict_label_prob(&mut self, text: &str, model: &Model, label_idx: usize) -> f32 {
        self.accumulator.fill(0.0);
        self.word_ids.clear();
        self.subword_buckets.clear();

        let dim = model.args.dim;
        let num_words = model.num_words;
        let bucket = model.args.bucket;

        // Split text on whitespace
        for token in text.split_whitespace() {
            if let Some(&wid) = model.word2id.get(token) {
                self.word_ids.push(wid);
            }
            if model.args.maxn > 0 {
                get_subword_hashes(
                    token,
                    model.args.minn,
                    model.args.maxn,
                    bucket,
                    &mut self.subword_buckets,
                );
            }
        }

        let total_features = self.word_ids.len() + self.subword_buckets.len();
        if total_features == 0 {
            return 0.0;
        }

        // Accumulate vocabulary word vectors
        for &wid in &self.word_ids {
            let offset = wid * dim;
            Self::add_row(&mut self.accumulator, &model.win[offset..offset + dim]);
        }

        // Accumulate subword vectors
        for &b in &self.subword_buckets {
            let offset = (num_words + b) * dim;
            Self::add_row(&mut self.accumulator, &model.win[offset..offset + dim]);
        }

        // Mean reduction across all participating tokens/subwords
        let scale = 1.0 / (total_features as f32);
        for v in self.accumulator.iter_mut() {
            *v *= scale;
        }

        if model.args.loss == 1 {
            // Hierarchical Softmax: walk binary tree path from root down to label leaf
            if label_idx >= model.paths.len() {
                return 0.0;
            }

            let path = &model.paths[label_idx];
            let code = &model.codes[label_idx];

            let mut prob = 1.0f32;
            for (&node_idx, &binary) in path.iter().zip(code.iter()) {
                let off = node_idx * dim;
                let logit = Self::dot_product(&self.accumulator, &model.wout[off..off + dim]);
                let s = Self::sigmoid(logit);
                if binary {
                    prob *= s;
                } else {
                    prob *= 1.0 - s;
                }
            }
            prob
        } else if model.args.loss == 3 {
            // Full Softmax
            let mut max_score = f32::NEG_INFINITY;
            let mut logits = vec![0.0f32; model.num_labels];

            for l in 0..model.num_labels {
                let off = l * dim;
                let s = Self::dot_product(&self.accumulator, &model.wout[off..off + dim]);
                logits[l] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for l in 0..model.num_labels {
                let exp_val = (logits[l] - max_score).exp();
                logits[l] = exp_val;
                sum_exp += exp_val;
            }

            if sum_exp > 0.0 {
                logits[label_idx] / sum_exp
            } else {
                0.0
            }
        } else {
            // Negative Sampling / Multi-label
            let wout_offset = label_idx * dim;
            let target_score = Self::dot_product(
                &self.accumulator,
                &model.wout[wout_offset..wout_offset + dim],
            );
            Self::sigmoid(target_score)
        }
    }
}
