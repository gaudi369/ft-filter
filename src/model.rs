use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, Cursor, Read};

pub const FASTTEXT_MAGIC: i32 = 0x2F4F16BA; // 793712314
pub const FASTTEXT_VERSION: i32 = 12;

#[derive(Debug, Clone)]
pub struct Args {
    pub dim: usize,
    pub ws: i32,
    pub epoch: i32,
    pub min_count: i32,
    pub neg: i32,
    pub word_ngrams: i32,
    pub loss: i32, // 1: hs (hierarchical softmax), 2: ns, 3: softmax
    pub model: i32, // 1: cbow, 2: sg, 3: supervised
    pub bucket: usize,
    pub minn: usize,
    pub maxn: usize,
    pub lr_update_rate: i32,
    pub t: f64,
}

#[derive(Clone, Default)]
pub struct Node {
    pub parent: i32,
    pub left: i32,
    pub right: i32,
    pub count: i64,
    pub binary: bool,
}

pub struct Model {
    pub args: Args,
    pub words: Vec<String>,
    pub labels: Vec<String>,
    pub label_counts: Vec<i64>,
    pub word2id: HashMap<String, usize>,
    pub label2id: HashMap<String, usize>,
    pub num_words: usize,
    pub num_labels: usize,
    pub win: Vec<f32>,  // Shape: (num_words + bucket) * dim
    pub wout: Vec<f32>, // Shape: num_labels * dim or (num_labels - 1) * dim for HS
    pub paths: Vec<Vec<usize>>, // For hierarchical softmax: path of internal node indices
    pub codes: Vec<Vec<bool>>,  // Binary branch decisions (true = right, false = left)
}

impl Model {
    /// Builds the exact binary Huffman tree matching FastText C++ Dictionary::initTree
    fn build_huffman_tree(label_counts: &[i64]) -> (Vec<Vec<usize>>, Vec<Vec<bool>>) {
        let k = label_counts.len();
        if k == 0 {
            return (Vec::new(), Vec::new());
        }
        if k == 1 {
            return (vec![vec![]], vec![vec![]]);
        }

        let mut tree = vec![Node::default(); 2 * k - 1];
        for i in 0..k {
            tree[i].count = label_counts[i];
            tree[i].parent = -1;
            tree[i].left = -1;
            tree[i].right = -1;
        }
        for i in k..(2 * k - 1) {
            tree[i].count = i64::MAX;
            tree[i].parent = -1;
            tree[i].left = -1;
            tree[i].right = -1;
        }

        let mut leaf = k as isize - 1;
        let mut node = k;

        for i in k..(2 * k - 1) {
            // Find two smallest nodes
            let mut mini = [0usize; 2];
            for j in 0..2 {
                if leaf >= 0 && tree[leaf as usize].count < tree[node].count {
                    mini[j] = leaf as usize;
                    leaf -= 1;
                } else {
                    mini[j] = node;
                    node += 1;
                }
            }

            tree[i].left = mini[0] as i32;
            tree[i].right = mini[1] as i32;
            tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
            tree[mini[0]].parent = i as i32;
            tree[mini[1]].parent = i as i32;
            tree[mini[1]].binary = true;
        }

        let mut paths = Vec::with_capacity(k);
        let mut codes = Vec::with_capacity(k);

        for i in 0..k {
            let mut path = Vec::new();
            let mut code = Vec::new();
            let mut j = i;
            while tree[j].parent >= 0 {
                let p = tree[j].parent as usize;
                // FastText internal node index in Wout is (parent - k)
                path.push(p - k);
                code.push(tree[j].binary);
                j = p;
            }
            paths.push(path);
            codes.push(code);
        }

        (paths, codes)
    }

    pub fn load_from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut r = Cursor::new(data);

        let magic = r.read_i32::<LittleEndian>()?;
        if magic != FASTTEXT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid FastText magic number: expected 0x{:X}, got 0x{:X}", FASTTEXT_MAGIC, magic),
            ));
        }

        let version = r.read_i32::<LittleEndian>()?;
        if version > FASTTEXT_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported model version"));
        }

        let args = Args {
            dim: r.read_i32::<LittleEndian>()? as usize,
            ws: r.read_i32::<LittleEndian>()?,
            epoch: r.read_i32::<LittleEndian>()?,
            min_count: r.read_i32::<LittleEndian>()?,
            neg: r.read_i32::<LittleEndian>()?,
            word_ngrams: r.read_i32::<LittleEndian>()?,
            loss: r.read_i32::<LittleEndian>()?,
            model: r.read_i32::<LittleEndian>()?,
            bucket: r.read_i32::<LittleEndian>()? as usize,
            minn: r.read_i32::<LittleEndian>()? as usize,
            maxn: r.read_i32::<LittleEndian>()? as usize,
            lr_update_rate: r.read_i32::<LittleEndian>()?,
            t: r.read_f64::<LittleEndian>()?,
        };

        let dict_size = r.read_i32::<LittleEndian>()? as usize;
        let _num_words = r.read_i32::<LittleEndian>()?;
        let _num_labels = r.read_i32::<LittleEndian>()?;
        let _ntokens = r.read_i64::<LittleEndian>()?;
        let pruneidx_size = r.read_i64::<LittleEndian>()?;

        if pruneidx_size > 0 {
            // Skip pruneidx table: pruneidx_size * 8 bytes (two i32)
            r.set_position(r.position() + (pruneidx_size as u64) * 8);
        }

        let mut words = Vec::new();
        let mut labels = Vec::new();
        let mut label_counts = Vec::new();
        let mut word2id = HashMap::new();
        let mut label2id = HashMap::new();

        for _ in 0..dict_size {
            let mut word_bytes = Vec::new();
            loop {
                let b = r.read_u8()?;
                if b == 0 {
                    break;
                }
                word_bytes.push(b);
            }
            let word = String::from_utf8_lossy(&word_bytes).to_string();
            let count = r.read_i64::<LittleEndian>()?;
            let entry_type = r.read_u8()?;

            if entry_type == 0 {
                let id = words.len();
                word2id.insert(word.clone(), id);
                words.push(word);
            } else if entry_type == 1 {
                let id = labels.len();
                label2id.insert(word.clone(), id);
                labels.push(word);
                label_counts.push(count);
            }
        }

        let (paths, codes) = if args.loss == 1 {
            Self::build_huffman_tree(&label_counts)
        } else {
            (Vec::new(), Vec::new())
        };

        let num_words = words.len();
        let num_labels = labels.len();

        // 1. Quant flag for Win
        let is_quant_in = r.read_u8()?;
        if is_quant_in != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Quantized Win not supported"));
        }

        let win_rows = r.read_i64::<LittleEndian>()? as usize;
        let win_cols = r.read_i64::<LittleEndian>()? as usize;
        let win_len = win_rows * win_cols;
        let mut win = vec![0.0f32; win_len];
        for val in win.iter_mut() {
            *val = r.read_f32::<LittleEndian>()?;
        }

        // 2. Quant flag for Wout
        let is_quant_out = r.read_u8()?;
        if is_quant_out != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Quantized Wout not supported"));
        }

        let wout_rows = r.read_i64::<LittleEndian>()? as usize;
        let wout_cols = r.read_i64::<LittleEndian>()? as usize;
        let wout_len = wout_rows * wout_cols;
        let mut wout = vec![0.0f32; wout_len];
        for val in wout.iter_mut() {
            *val = r.read_f32::<LittleEndian>()?;
        }

        Ok(Model {
            args,
            words,
            labels,
            label_counts,
            word2id,
            label2id,
            num_words,
            num_labels,
            win,
            wout,
            paths,
            codes,
        })
    }
}
