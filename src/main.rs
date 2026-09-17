mod infer;
mod model;
mod tokenize;

use clap::{Args, Parser, Subcommand};
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ft-filter",
    version,
    about = "High-throughput CPU FastText engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Filter or annotate text using a pre-trained FastText binary model
    Filter(FilterArgs),
    /// Train a new FastText binary model
    Train(TrainArgs),
}

#[derive(Args, Debug, Clone)]
pub struct FilterArgs {
    #[arg(short = 'm', long = "model", value_name = "PATH")]
    pub model: PathBuf,

    #[arg(short = 'i', long = "input", value_name = "PATH")]
    pub input: Option<PathBuf>,

    #[arg(short = 'o', long = "output", value_name = "PATH")]
    pub output: Option<PathBuf>,

    #[arg(short = 'f', long = "format", default_value = "jsonl")]
    pub format: String,

    #[arg(short = 'k', long = "json-key", default_value = "text")]
    pub json_key: String,

    #[arg(short = 't', long = "threshold", default_value_t = 0.5)]
    pub threshold: f32,

    #[arg(short = 'v', long = "invert", default_value_t = false)]
    pub invert: bool,

    #[arg(short = 's', long = "emit-score", default_value_t = false)]
    pub emit_score: bool,

    #[arg(long = "score-key", default_value = "ft_score")]
    pub score_key: String,

    #[arg(short = 'l', long = "label")]
    pub label: Option<String>,

    /// Print timing and workload statistics as JSON to stderr
    #[arg(long = "stats", default_value_t = false)]
    pub stats: bool,
}

#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    #[arg(short = 'i', long = "input", value_name = "PATH")]
    pub input: PathBuf,

    #[arg(short = 'o', long = "output", value_name = "PATH")]
    pub output: PathBuf,

    #[arg(long = "dim", default_value_t = 256)]
    pub dim: usize,

    #[arg(long = "lr", default_value_t = 0.1)]
    pub lr: f32,

    #[arg(long = "epochs", default_value_t = 5)]
    pub epochs: usize,

    #[arg(long = "bucket", default_value_t = 2_000_000)]
    pub bucket: usize,

    #[arg(long = "minn", default_value_t = 3)]
    pub minn: usize,

    #[arg(long = "maxn", default_value_t = 6)]
    pub maxn: usize,

    #[arg(long = "min-count", default_value_t = 5)]
    pub min_count: u64,
}

fn run_filter(args: FilterArgs) -> io::Result<()> {
    let load_start = std::time::Instant::now();
    let model_file = File::open(&args.model)?;
    // Streaming load: matrices are read directly from the file, so the mmap
    // path (which doubled peak RSS) is no longer used.
    let model = model::Model::load_from_reader(model_file)?;
    let mut stats = Stats {
        load_seconds: load_start.elapsed().as_secs_f64(),
        ..Stats::default()
    };

    let target_label = args.label.clone().unwrap_or_else(|| {
        model
            .labels
            .first()
            .cloned()
            .expect("Model contains no output labels")
    });
    let label_idx = *model
        .label2id
        .get(&target_label)
        .unwrap_or_else(|| panic!("Label '{}' not found in model", target_label));

    let reader: Box<dyn BufRead> = match &args.input {
        Some(path) => Box::new(BufReader::with_capacity(1024 * 1024, File::open(path)?)),
        None => Box::new(BufReader::with_capacity(1024 * 1024, io::stdin())),
    };

    let mut writer: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(BufWriter::with_capacity(1024 * 1024, File::create(path)?)),
        None => Box::new(BufWriter::with_capacity(1024 * 1024, io::stdout())),
    };

    let mut session = infer::InferSession::new(model.args.dim);
    let mut line_buf = String::with_capacity(16384);
    let mut reader = reader;
    let is_jsonl = args.format == "jsonl";
    let process_start = std::time::Instant::now();

    while reader.read_line(&mut line_buf)? > 0 {
        if args.stats {
            stats.input_bytes += line_buf.len() as u64;
        }
        // Remove record framing only: Unicode whitespace can be part of a token.
        let trimmed_line = line_buf.trim_end_matches(['\r', '\n']);
        if trimmed_line.is_empty() {
            line_buf.clear();
            continue;
        }

        if args.stats {
            stats.records += 1;
        }
        let mut parsed_obj = None;
        let prob = if is_jsonl {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed_line) {
                let score = if let Some(text) = map.get(&args.json_key).and_then(|v| v.as_str()) {
                    session.predict_label_prob(text, &model, label_idx)
                } else {
                    0.0
                };
                if args.emit_score {
                    parsed_obj = Some(map);
                }
                score
            } else {
                0.0
            }
        } else {
            session.predict_label_prob(trimmed_line, &model, label_idx)
        };

        let passes = if args.invert {
            prob < args.threshold
        } else {
            prob >= args.threshold
        };

        if passes {
            if args.stats {
                stats.passed_records += 1;
            }
            if args.emit_score {
                if is_jsonl {
                    if let Some(mut map) = parsed_obj {
                        map.insert(args.score_key.clone(), serde_json::json!(prob));
                        serde_json::to_writer(&mut writer, &map)?;
                        writer.write_all(b"\n")?;
                    }
                } else {
                    writeln!(writer, "{:.5}\t{}", prob, trimmed_line)?;
                }
            } else {
                writer.write_all(trimmed_line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
        }

        line_buf.clear();
    }

    writer.flush()?;
    stats.process_seconds = process_start.elapsed().as_secs_f64();
    if args.stats {
        serde_json::to_writer(io::stderr().lock(), &stats)?;
        eprintln!();
    }
    Ok(())
}

#[derive(serde::Serialize, Default)]
struct Stats {
    records: u64,
    input_bytes: u64,
    passed_records: u64,
    process_seconds: f64,
    load_seconds: f64,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Filter(args) => run_filter(args),
        Commands::Train(_) => {
            eprintln!("Training implementation will be hooked here.");
            Ok(())
        }
    }
}
