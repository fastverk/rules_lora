//! runpod_orchestrator: write a `lora.v1.TrainingJobSpec` to disk
//! at build time, run a remote LoRA training job at execute time.
//!
//! v0.0.1 scope: only the `write-jobspec` subcommand. The full
//! `run` subcommand (upload + poll + download) lands in v0.1 once
//! we lift runpod-cli's reusable bits out of prime-transformer.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "runpod_orchestrator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Build-time: serialize the rule attrs into a TrainingJobSpec
    /// JSON file. The execute-time `run` subcommand reads this.
    WriteJobspec {
        #[arg(long)]
        name: String,
        #[arg(long)]
        recipe: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        base_id: String,
        #[arg(long)]
        base_revision: String,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Execute-time: upload spec + dataset, poll the job, download
    /// the adapter. Not yet implemented; v0.1.
    Run {
        #[arg(long)]
        jobspec: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::WriteJobspec {
            name,
            recipe,
            dataset,
            base_id,
            base_revision,
            backend,
            out,
        } => write_jobspec(name, recipe, dataset, base_id, base_revision, backend, out),
        Cmd::Run { jobspec } => {
            anyhow::bail!(
                "runpod_orchestrator run: not implemented yet (v0.1). \
                 v0.0 only emits the spec at {}.",
                jobspec.display()
            )
        }
    }
}

fn write_jobspec(
    name: String,
    recipe: PathBuf,
    dataset: PathBuf,
    base_id: String,
    base_revision: String,
    backend: String,
    out: PathBuf,
) -> Result<()> {
    let recipe_bytes =
        std::fs::read(&recipe).with_context(|| format!("reading {}", recipe.display()))?;
    let dataset_bytes =
        std::fs::read(&dataset).with_context(|| format!("reading {}", dataset.display()))?;
    let recipe_sha = blake3::hash(&recipe_bytes).to_hex().to_string();
    let dataset_sha = blake3::hash(&dataset_bytes).to_hex().to_string();
    let recipe_yaml = String::from_utf8(recipe_bytes).context("recipe is not utf-8")?;

    let spec = serde_json::json!({
        "name": name,
        "recipe_sha": recipe_sha,
        "dataset_sha": dataset_sha,
        "base_model_id": base_id,
        "base_model_revision": base_revision,
        "backend": backend,
        "recipe_yaml": recipe_yaml,
        "backend_config_json": "{}",
        "max_minutes": 0,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&spec)?)
        .with_context(|| format!("writing {}", out.display()))?;
    eprintln!(
        "runpod_orchestrator: jobspec → {} (recipe={}…, dataset={}…)",
        out.display(),
        &recipe_sha[..12],
        &dataset_sha[..12]
    );
    Ok(())
}
