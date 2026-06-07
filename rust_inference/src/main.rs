use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use tch::{no_grad, CModule, Device, Kind, Tensor};

const FEATURE_COUNT: usize = 16;

struct Args {
    model_path: String,
    input_path: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let sequence = read_csv_sequence(&args.input_path)?;
    let sequence_length = sequence.len() / FEATURE_COUNT;
    let input = Tensor::from_slice(&sequence)
        .reshape([1, sequence_length as i64, FEATURE_COUNT as i64])
        .to_kind(Kind::Float);

    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let model = CModule::load_on_device(&args.model_path, device)?;
    let output = no_grad(|| model.forward_ts(&[input.to_device(device)]))?;
    let predictions = output.flatten(0, -1).to_device(Device::Cpu);

    for index in 0..predictions.numel() {
        println!("{}", predictions.double_value(&[index as i64]));
    }

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut model_path = String::from("../models/gru_model_torchscript.pt");
    let mut input_path = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_path = args.next().ok_or("--model requires a path")?;
            }
            "--input" => {
                input_path = Some(args.next().ok_or("--input requires a path")?);
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

    Ok(Args {
        model_path,
        input_path,
    })
}

fn print_help() {
    println!(
        "Usage: energy-gru-inference --model ../models/gru_model_torchscript.pt --input window.csv"
    );
    println!();
    println!("The input CSV must contain one sequence window with 16 numeric features per row.");
    println!("A header row is allowed and will be skipped if it is not numeric.");
}

fn read_csv_sequence(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
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
