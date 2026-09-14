mod infer;
mod model;
mod tokenize;

use clap::Parser;
use memmap2::Mmap;
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Lean single-threaded FastText CPU filter", long_about = None)]
struct Cli {
    #[arg(short = 'm', long = "model", value_name = "PATH")]
    model: PathBuf,

    #[arg(short = 'i', long = "input", value_name = "PATH")]
    input: Option<PathBuf>,

    #[arg(short = 'o', long = "output", value_name = "PATH")]
    output: Option<PathBuf>,

    #[arg(short = 'f', long = "format", default_value = "jsonl")]
    format: String,

    #[arg(short = 'k', long = "json-key", default_value = "text")]
    json_key: String,

    #[arg(short = 't', long = "threshold", default_value_t = 0.5)]
    threshold: f32,

    #[arg(short = 'l', long = "label")]
    label: Option<String>,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    // Map the model file into memory (zero-copy, demand-paged).
    let model_file = File::open(&cli.model)?;
    let mmap = unsafe { Mmap::map(&model_file)? };
    let model = model::Model::load_from_bytes(&mmap)?;

    // Determine target label (auto-detect first if unspecified).
    let target_label = cli
        .label
        .clone()
        .unwrap_or_else(|| {
            model
                .labels
                .first()
                .cloned()
                .expect("Model contains no output labels")
        });
    let label_idx = *model
        .label2id
        .get(&target_label)
        .expect("Target label not found in model");

    // Input reader
    let reader: Box<dyn BufRead> = match cli.input {
        Some(path) => Box::new(BufReader::with_capacity(
            1024 * 1024,
            File::open(path)?,
        )),
        None => Box::new(BufReader::with_capacity(1024 * 1024, io::stdin())),
    };

    // Output writer
    let mut writer: Box<dyn Write> = match cli.output {
        Some(path) => Box::new(BufWriter::with_capacity(
            1024 * 1024,
            File::create(path)?,
        )),
        None => Box::new(BufWriter::with_capacity(1024 * 1024, io::stdout())),
    };

    let mut session = infer::InferSession::new(model.args.dim);
    let mut line_buf = String::with_capacity(8192);
    let mut reader = reader;

    let is_jsonl = cli.format == "jsonl";

    while reader.read_line(&mut line_buf)? > 0 {
        let prob = if is_jsonl {
            match serde_json::from_str::<Value>(&line_buf) {
                Ok(Value::Object(map)) => {
                    if let Some(Value::String(text)) = map.get(&cli.json_key) {
                        session.predict_label_prob(text, &model, label_idx)
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            }
        } else {
            session.predict_label_prob(line_buf.trim_end(), &model, label_idx)
        };

        if prob >= cli.threshold {
            writer.write_all(line_buf.as_bytes())?;
        }

        line_buf.clear();
    }

    writer.flush()?;
    Ok(())
}
