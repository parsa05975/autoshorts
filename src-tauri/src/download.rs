//! Downloading streams and videos from a link with yt-dlp, with live progress and cancel.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// Marks yt-dlp's progress lines so they can be told apart from everything else it prints.
const PROGRESS_TAG: &str = "AS_PROGRESS";

/// 1080p is plenty for a 1080x1920 clip; higher qualities make multi-hour streams
/// several times bigger and slower to download. H.264 is preferred because it decodes fastest.
const FORMAT: &str = "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b";

/// Process id of the running download, so it can be cancelled.
static ACTIVE_PID: Mutex<Option<u32>> = Mutex::new(None);
static CANCELLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    /// "video", "audio" or "file" (a single stream with both).
    pub part: &'static str,
    pub downloaded_bytes: Option<f64>,
    pub total_bytes: Option<f64>,
    pub speed_bytes_per_sec: Option<f64>,
    pub eta_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkInfo {
    pub license: Option<String>,
    pub is_safe: bool,
    pub title: Option<String>,
    pub duration_sec: Option<f64>,
}

fn ytdlp() -> Command {
    let mut command = Command::new("yt-dlp");
    hide_window(&mut command);
    command
}

#[cfg(windows)]
fn hide_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_window(_command: &mut Command) {}

/// Reads the link's details and refuses streams that are still live or still being processed.
pub fn check_link(url: &str) -> Result<LinkInfo, String> {
    let output = ytdlp()
        .args(["--dump-json", "--no-playlist", "--no-warnings", url])
        .output()
        .map_err(|e| format!("Failed to run yt-dlp: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Couldn't read that link. {}", last_error_line(&stderr)));
    }
    let json = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = json
        .lines()
        .find_map(|line| serde_json::from_str(line).ok())
        .ok_or_else(|| "Failed to parse yt-dlp output".to_string())?;
    link_info_from_json(&parsed)
}

fn link_info_from_json(parsed: &serde_json::Value) -> Result<LinkInfo, String> {
    match parsed.get("live_status").and_then(|v| v.as_str()) {
        Some("is_live") => {
            return Err(
                "This stream is still live. Wait until it ends and the full video (VOD) is up, then import it."
                    .to_string(),
            )
        }
        Some("is_upcoming") => return Err("This stream hasn't started yet.".to_string()),
        Some("post_live") => {
            return Err("This stream just ended and YouTube is still processing it. \
                Try again once the video shows its full length (usually within a few hours)."
                .to_string())
        }
        _ => {}
    }
    let license = parsed.get("license").and_then(|v| v.as_str()).map(str::to_string);
    let is_safe = license.as_deref().is_some_and(|lic| {
        let lic = lic.to_lowercase();
        lic.contains("creative commons") || lic.contains("reuse allowed")
    });
    Ok(LinkInfo {
        license,
        is_safe,
        title: parsed.get("title").and_then(|v| v.as_str()).map(str::to_string),
        duration_sec: parsed.get("duration").and_then(|v| v.as_f64()),
    })
}

fn progress_template() -> String {
    format!(
        "download:{PROGRESS_TAG} %(info.vcodec)s %(info.acodec)s %(progress.downloaded_bytes)s \
         %(progress.total_bytes)s %(progress.total_bytes_estimate)s %(progress.speed)s %(progress.eta)s"
    )
}

fn number(field: &str) -> Option<f64> {
    field.parse::<f64>().ok().filter(|n| n.is_finite() && *n >= 0.0)
}

/// Parses one line written by [`progress_template`]. Returns None for any other line.
pub fn parse_progress_line(line: &str) -> Option<DownloadProgress> {
    let rest = line.trim().strip_prefix(PROGRESS_TAG)?;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    if fields.len() != 7 {
        return None;
    }
    let has_video = !matches!(fields[0], "none" | "NA");
    let has_audio = !matches!(fields[1], "none" | "NA");
    let part = match (has_video, has_audio) {
        (true, false) => "video",
        (false, true) => "audio",
        _ => "file",
    };
    Some(DownloadProgress {
        part,
        downloaded_bytes: number(fields[2]),
        total_bytes: number(fields[3]).or_else(|| number(fields[4])),
        speed_bytes_per_sec: number(fields[5]),
        eta_sec: number(fields[6]),
    })
}

fn last_error_line(stderr: &str) -> String {
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with("ERROR"))
        .or_else(|| stderr.lines().rev().map(str::trim).find(|line| !line.is_empty()))
        .unwrap_or("")
        .to_string()
}

/// Downloads `url` into `out_dir`, calling `on_progress` as it goes, and returns the saved file.
/// A cancelled download leaves its partial file in place, so the next try resumes it.
pub fn download(
    url: &str,
    out_dir: &Path,
    on_progress: impl Fn(DownloadProgress) + Send + Sync + 'static,
) -> Result<PathBuf, String> {
    let template = out_dir.join("AutoShorts_%(id)s.%(ext)s");
    let mut child = ytdlp()
        .args(["--newline", "--progress", "--no-playlist", "--encoding", "utf-8"])
        .args(["--concurrent-fragments", "8"])
        .args(["--format", FORMAT, "--merge-output-format", "mp4"])
        .arg("--output")
        .arg(&template)
        .args(["--progress-template", &progress_template()])
        .args(["--print", "after_move:filepath", "--no-simulate"])
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run yt-dlp: {e}"))?;

    CANCELLED.store(false, Ordering::SeqCst);
    *ACTIVE_PID.lock().unwrap() = Some(child.id());

    let on_progress = Arc::new(on_progress);
    let stderr = child.stderr.take().expect("stderr is piped");
    let stderr_reader = {
        let on_progress = Arc::clone(&on_progress);
        std::thread::spawn(move || {
            let mut text = String::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                match parse_progress_line(&line) {
                    Some(progress) => on_progress(progress),
                    None => {
                        text.push_str(line.trim_end());
                        text.push('\n');
                    }
                }
            }
            text
        })
    };

    let mut saved: Option<PathBuf> = None;
    let stdout = child.stdout.take().expect("stdout is piped");
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Some(progress) = parse_progress_line(&line) {
            on_progress(progress);
            continue;
        }
        let candidate = PathBuf::from(line.trim());
        if !line.trim().is_empty() && candidate.is_file() {
            saved = Some(candidate);
        }
    }

    let status = child.wait().map_err(|e| format!("yt-dlp stopped unexpectedly: {e}"))?;
    *ACTIVE_PID.lock().unwrap() = None;
    let stderr_text = stderr_reader.join().unwrap_or_default();

    if CANCELLED.swap(false, Ordering::SeqCst) {
        return Err("Download cancelled.".to_string());
    }
    if !status.success() {
        return Err(format!("Download failed. {}", last_error_line(&stderr_text)));
    }
    saved.ok_or_else(|| "Download finished but the file couldn't be found in your Downloads folder.".to_string())
}

/// Stops the running download, if there is one. Returns whether one was running.
pub fn cancel() -> bool {
    let Some(pid) = *ACTIVE_PID.lock().unwrap() else {
        return false;
    };
    CANCELLED.store(true, Ordering::SeqCst);
    #[cfg(windows)]
    {
        // /T also stops the ffmpeg and helper processes yt-dlp started.
        let mut kill = Command::new("taskkill");
        hide_window(&mut kill);
        let _ = kill.args(["/PID", &pid.to_string(), "/T", "/F"]).output();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").arg(pid.to_string()).output();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_video_progress() {
        let line = "AS_PROGRESS avc1.640028 none 1048576 8388608 NA 2097152.5 3";
        assert_eq!(
            parse_progress_line(line),
            Some(DownloadProgress {
                part: "video",
                downloaded_bytes: Some(1048576.0),
                total_bytes: Some(8388608.0),
                speed_bytes_per_sec: Some(2097152.5),
                eta_sec: Some(3.0),
            })
        );
    }

    #[test]
    fn falls_back_to_estimated_total_and_handles_missing_values() {
        let progress = parse_progress_line("AS_PROGRESS none mp4a.40.2 500 NA 1000.0 NA NA\r").unwrap();
        assert_eq!(progress.part, "audio");
        assert_eq!(progress.total_bytes, Some(1000.0));
        assert_eq!(progress.speed_bytes_per_sec, None);
        assert_eq!(progress.eta_sec, None);
    }

    #[test]
    fn ignores_other_lines() {
        assert_eq!(parse_progress_line("[youtube] abc: Downloading webpage"), None);
        assert_eq!(parse_progress_line("C:\\Users\\me\\Downloads\\AutoShorts_abc.mp4"), None);
    }

    #[test]
    fn refuses_live_and_processing_streams() {
        for status in ["is_live", "is_upcoming", "post_live"] {
            let json = serde_json::json!({ "live_status": status });
            assert!(link_info_from_json(&json).is_err(), "{status}");
        }
        let json = serde_json::json!({ "live_status": "was_live", "duration": 25200.0, "license": null });
        let info = link_info_from_json(&json).unwrap();
        assert_eq!(info.duration_sec, Some(25200.0));
        assert!(!info.is_safe);
    }

    #[test]
    fn picks_the_error_line() {
        let stderr = "WARNING: something\nERROR: [youtube] abc: Video unavailable\n";
        assert_eq!(last_error_line(stderr), "ERROR: [youtube] abc: Video unavailable");
    }
}
