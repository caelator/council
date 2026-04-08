use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use substrate::blob::{BlobEnvelope, BlobStore, ContentType, ProducerId};

use crate::trace::append_jsonl;

fn blob_store_path() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| anyhow::anyhow!("HOME not set"))?;
    Ok(PathBuf::from(home).join(".openclaw/blobs"))
}

/// Dual-write: append to JSONL file, then also write the same payload to the blob store.
/// Blob write failures are logged but do not fail the operation.
pub fn dual_write_jsonl(
    jsonl_path: &Path,
    value: &impl Serialize,
    content_type: &str,
) -> Result<()> {
    // Always write to JSONL first (the primary path).
    append_jsonl(jsonl_path, value)?;

    // Best-effort blob write.
    if let Err(e) = write_blob(value, content_type) {
        eprintln!("  [warn] blob dual-write failed for {content_type}: {e}");
    }

    Ok(())
}

fn write_blob(value: &impl Serialize, content_type: &str) -> Result<(), anyhow::Error> {
    let blob_dir = blob_store_path()?;

    let mut store = BlobStore::open(&blob_dir)
        .map_err(|e| anyhow::anyhow!("blob store open: {e}"))?;

    let payload = serde_json::to_vec(value)?;
    let ct = ContentType::new(content_type)
        .map_err(|e| anyhow::anyhow!("content type: {e}"))?;
    let producer = ProducerId::new("council")
        .map_err(|e| anyhow::anyhow!("producer id: {e}"))?;

    let envelope = BlobEnvelope::new(ct, producer, payload, None, BTreeMap::new());
    store
        .put(&envelope)
        .map_err(|e| anyhow::anyhow!("blob put: {e}"))?;

    Ok(())
}
