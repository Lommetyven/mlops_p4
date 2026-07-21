use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use tch::{no_grad, CModule, Device, Kind, Tensor};

const FEATURE_COUNT: usize = 16;

struct Args {
    model_path: String,
    input_path: String,
    sequence_length: usize,
    batch_size: usize,
    precision: Precision,
}

#[derive(Clone, Copy)]
enum Precision {
    Fp32,
    Fp16,
    Fp8,
}

impl Precision {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "fp32" => Ok(Self::Fp32),
            "fp16" => Ok(Self::Fp16),
            "fp8" => Ok(Self::Fp8),
            _ => Err("--precision must be one of fp32, fp16, or fp8".into()),
        }
    }

    fn kind(self) -> Result<Kind, Box<dyn Error>> {
        match self {
            Self::Fp32 => Ok(Kind::Float),
            Self::Fp16 => Ok(Kind::Half),
            Self::Fp8 => Err(
                "FP8 inference is not supported by the current TorchScript GRU/tch 0.19 backend; select FP16 or FP32"
                    .into(),
            ),
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let precision_kind = args.precision.kind()?;
    let rows = read_csv_rows(&args.input_path)?;
    let row_count = rows.len() / FEATURE_COUNT;
    if row_count < args.sequence_length {
        return Err(format!(
            "input CSV has {row_count} rows but sequence length is {}",
            args.sequence_length
        )
        .into());
    }

    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    if matches!(args.precision, Precision::Fp16) && matches!(device, Device::Cpu) {
        return Err("FP16 inference requires a CUDA GPU; select FP32 on CPU".into());
    }

    let mut model = CModule::load_on_device(&args.model_path, device)?;
    model.set_eval();
    model.to(device, precision_kind, false);

    let window_count = row_count - args.sequence_length + 1;
    for batch_start in (0..window_count).step_by(args.batch_size) {
        let batch_end = (batch_start + args.batch_size).min(window_count);
        let batch_count = batch_end - batch_start;
        let mut batch =
            Vec::with_capacity(batch_count * args.sequence_length * FEATURE_COUNT);

        for window_start in batch_start..batch_end {
            let value_start = window_start * FEATURE_COUNT;
            let value_end = (window_start + args.sequence_length) * FEATURE_COUNT;
            batch.extend_from_slice(&rows[value_start..value_end]);
        }

        let input = Tensor::from_slice(&batch)
            .reshape([
                batch_count as i64,
                args.sequence_length as i64,
                FEATURE_COUNT as i64,
            ])
            .to_device(device)
            .to_kind(precision_kind);
        let output = no_grad(|| model.forward_ts(&[input]))?;
        let predictions = output
            .flatten(0, -1)
            .to_device(Device::Cpu)
            .to_kind(Kind::Float);

        for index in 0..predictions.numel() {
            println!("{}", predictions.double_value(&[index as i64]));
        }
    }

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut model_path = String::from("../models/gru_model_torchscript.pt");
    let mut input_path = None;
    let mut sequence_length = 12usize;
    let mut batch_size = 1024usize;
    let mut precision = Precision::Fp32;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_path = args.next().ok_or("--model requires a path")?;
            }
            "--input" => {
                input_path = Some(args.next().ok_or("--input requires a path")?);
            }
            "--sequence-length" => {
                sequence_length = args
                    .next()
                    .ok_or("--sequence-length requires a value")?
                    .parse()?;
            }
            "--batch-size" => {
                batch_size = args
                    .next()
                    .ok_or("--batch-size requires a value")?
                    .parse()?;
            }
            "--precision" => {
                precision = Precision::parse(
                    &args.next().ok_or("--precision requires a value")?,
                )?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}").into()),
        }
    }

    let input_path = input_path.ok_or("--input is required")?;
    if !Path::new(&model_path).is_file() {
        return Err(format!("model not found: {model_path}").into());
    }
    if !Path::new(&input_path).is_file() {
        return Err(format!("input CSV not found: {input_path}").into());
    }
    if sequence_length == 0 {
        return Err("--sequence-length must be greater than zero".into());
    }
    if batch_size == 0 {
        return Err("--batch-size must be greater than zero".into());
    }

    Ok(Args {
        model_path,
        input_path,
        sequence_length,
        batch_size,
        precision,
    })
}

fn print_help() {
    println!(
        "Usage: energy-gru-inference --model MODEL.pt --input DATA.csv [--sequence-length 12] [--batch-size 1024] [--precision fp32|fp16|fp8]"
    );
    println!();
    println!("Every overlapping sequence window in the input CSV is inferred.");
    println!("FP16 requires CUDA. FP8 is rejected because TorchScript GRU does not support it.");
    println!("The input CSV must contain 16 numeric features per row.");
    println!("A header row is allowed and will be skipped if it is not numeric.");
}

fn read_csv_rows(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let mut values = Vec::new();
    let mut parsed_rows = 0usize;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').map(str::trim).collect();
        if fields.len() != FEATURE_COUNT {
            if parsed_rows == 0 && fields.iter().any(|field| field.parse::<f32>().is_err()) {
                continue;
            }
            return Err(format!(
                "expected {FEATURE_COUNT} columns, got {} in line: {trimmed}",
                fields.len()
            )
            .into());
        }

        let mut row = Vec::with_capacity(FEATURE_COUNT);
        let mut non_numeric = false;
        for field in fields {
            match field.parse::<f32>() {
                Ok(value) => row.push(value),
                Err(_) => {
                    non_numeric = true;
                    break;
                }
            }
        }

        if non_numeric {
            if parsed_rows == 0 {
                continue;
            }
            return Err(format!("non-numeric value in line: {trimmed}").into());
        }

        values.extend(row);
        parsed_rows += 1;
    }

    if parsed_rows == 0 {
        return Err("input CSV did not contain any numeric sequence rows".into());
    }

    Ok(values)
}
