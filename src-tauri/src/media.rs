use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::models::{ClipLayout, LoudnessSpike, MediaProbe, NormRect};

pub fn command_exists(name: &str) -> bool {
    Command::new(name).arg("-version").output().is_ok()
}

pub fn probe_media(path: &str) -> Result<MediaProbe> {
    if !command_exists("ffprobe") {
        return Err(anyhow!("ffprobe is not installed or not available on PATH"));
    }

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()
        .context("running ffprobe")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let json: Value = serde_json::from_slice(&output.stdout).context("parsing ffprobe JSON")?;
    let streams = json
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let video = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"));
    let audio = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("audio"));

    let duration_sec = json
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(Value::as_str)
        .and_then(|duration| duration.parse::<f64>().ok());

    Ok(MediaProbe {
        duration_sec,
        has_video: video.is_some(),
        width: video
            .and_then(|stream| stream.get("width"))
            .and_then(Value::as_i64),
        height: video
            .and_then(|stream| stream.get("height"))
            .and_then(Value::as_i64),
        video_codec: video
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        audio_codec: audio
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

pub fn extract_audio(source_path: &str, project_dir: &Path) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    std::fs::create_dir_all(project_dir)?;
    let output_path = project_dir.join("transcription_audio.wav");

    let output = Command::new("ffmpeg")
        .args(["-y", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000"])
        .arg(&output_path)
        .output()
        .context("running ffmpeg audio extraction")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg audio extraction failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    Ok(output_path)
}

pub fn render_flat_clip(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    output_path: &Path,
    drawtext_filters: Option<&str>,
    layout: &ClipLayout,
) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let start = format!("{start_sec:.3}");
    let duration_sec = (end_sec - start_sec).max(0.1);
    let duration = format!("{duration_sec:.3}");

    let probe = probe_media(source_path).ok();
    let has_video = probe.as_ref().map(|p| p.has_video).unwrap_or(false);

    let streamer_facecam = match layout {
        ClipLayout::Streamer { facecam } if has_video => Some(*facecam),
        _ => None,
    };

    // The video filter graph. Captions add one filter per word, so it is written to a
    // file instead of the command line: Windows caps a command line at ~32,000 chars.
    let graph = if let Some(facecam) = streamer_facecam {
        let (src_w, src_h) = match probe.as_ref().and_then(|p| p.width.zip(p.height)) {
            Some((w, h)) if w > 0 && h > 0 => (w, h),
            _ => {
                return Err(anyhow!(
                    "streamer layout needs the video's size, but ffprobe could not read it"
                ))
            }
        };
        Some(streamer_filter_graph(src_w, src_h, &facecam, drawtext_filters))
    } else if has_video {
        let mut filter = "crop=w='2*trunc(min(iw,ih*9/16)/2)':h='2*trunc(min(ih,iw*16/9)/2)'".to_string();
        if let Some(drawtext) = drawtext_filters.filter(|d| !d.is_empty()) {
            filter = format!("{filter},{drawtext}");
        }
        Some(format!("[0:v]{filter}[v]"))
    } else {
        None
    };

    let graph_file = output_path.with_extension("filtergraph.txt");
    if let Some(graph) = &graph {
        std::fs::write(&graph_file, graph).context("writing ffmpeg filter graph")?;
    }

    let run = |graph_option: &str| -> Result<std::process::Output> {
        let mut cmd = Command::new("ffmpeg");
        // Only real errors on stderr, so a failure message shows the cause, not FFmpeg's banner.
        cmd.args(["-hide_banner", "-loglevel", "error"]);
        cmd.args(["-y", "-ss", &start, "-i", source_path, "-t", &duration]);
        if graph.is_some() {
            cmd.arg(graph_option).arg(&graph_file);
            cmd.args(["-map", "[v]", "-map", "0:a?"]);
            cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]);
        } else {
            cmd.arg("-vn");
        }
        cmd.args(["-c:a", "aac", "-b:a", "192k"]);
        cmd.arg(output_path);
        cmd.output().context("running ffmpeg clip render")
    };

    // FFmpeg 7+ reads an option's value from a file with "-/option"; older versions only
    // know -filter_complex_script (which newer ones drop). Try the new form, then the old.
    let mut output = run("-/filter_complex")?;
    if !output.status.success() && String::from_utf8_lossy(&output.stderr).contains("Unrecognized option") {
        output = run("-filter_complex_script")?;
    }
    let _ = std::fs::remove_file(&graph_file);

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg clip render failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    Ok(output_path.to_path_buf())
}

/// Output size of a streamer-layout clip (9:16).
pub const STREAMER_OUT_W: i64 = 1080;
pub const STREAMER_OUT_H: i64 = 1920;
/// Facecam band height: one third of the output. Gameplay fills the remaining two thirds.
pub const STREAMER_FACECAM_H: i64 = 640;
pub const STREAMER_GAMEPLAY_H: i64 = STREAMER_OUT_H - STREAMER_FACECAM_H;

/// A crop in source pixels, as FFmpeg's `crop=w:h:x:y` expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelCrop {
    pub w: i64,
    pub h: i64,
    pub x: i64,
    pub y: i64,
}

fn even_floor(value: f64) -> i64 {
    ((value / 2.0).floor() as i64 * 2).max(2)
}

/// Turns a normalized rectangle into a pixel crop with exactly the target aspect ratio.
///
/// The rectangle is clamped into the frame, then trimmed (keeping its center) so its
/// aspect matches `aspect_w:aspect_h`. Nothing is stretched, so what the user picks is
/// what ends up on screen, minus any trim needed to match the band's shape.
pub fn fit_crop_to_aspect(
    src_w: i64,
    src_h: i64,
    rect: &NormRect,
    aspect_w: i64,
    aspect_h: i64,
) -> PixelCrop {
    let fw = src_w as f64;
    let fh = src_h as f64;
    let sane = |v: f64, fallback: f64| if v.is_finite() { v } else { fallback };

    let x0 = sane(rect.x, 0.0).clamp(0.0, 1.0) * fw;
    let y0 = sane(rect.y, 0.0).clamp(0.0, 1.0) * fh;
    let x1 = (sane(rect.x, 0.0) + sane(rect.w, 1.0)).clamp(0.0, 1.0) * fw;
    let y1 = (sane(rect.y, 0.0) + sane(rect.h, 1.0)).clamp(0.0, 1.0) * fh;

    // Degenerate selections fall back to the whole frame rather than failing the render.
    let (x0, y0, x1, y1) = if x1 - x0 < 2.0 || y1 - y0 < 2.0 {
        (0.0, 0.0, fw, fh)
    } else {
        (x0, y0, x1, y1)
    };

    let target = aspect_w as f64 / aspect_h as f64;
    let (mut w, mut h) = (x1 - x0, y1 - y0);
    if w / h > target {
        w = h * target;
    } else {
        h = w / target;
    }

    let cx = (x0 + x1) / 2.0;
    let cy = (y0 + y1) / 2.0;
    let w_px = even_floor(w).min(src_w - src_w % 2);
    let h_px = even_floor(h).min(src_h - src_h % 2);
    let x = ((cx - w_px as f64 / 2.0).round() as i64).clamp(0, src_w - w_px);
    let y = ((cy - h_px as f64 / 2.0).round() as i64).clamp(0, src_h - h_px);

    PixelCrop { w: w_px, h: h_px, x, y }
}

/// Builds the `-filter_complex` graph that stacks the facecam over the gameplay.
/// The final video pad is labelled `[v]`.
pub fn streamer_filter_graph(
    src_w: i64,
    src_h: i64,
    facecam: &NormRect,
    drawtext_filters: Option<&str>,
) -> String {
    let cam = fit_crop_to_aspect(src_w, src_h, facecam, STREAMER_OUT_W, STREAMER_FACECAM_H);
    let whole_frame = NormRect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
    let game = fit_crop_to_aspect(src_w, src_h, &whole_frame, STREAMER_OUT_W, STREAMER_GAMEPLAY_H);

    let mut graph = format!(
        "[0:v]split=2[cam_src][game_src];\
         [cam_src]crop={cw}:{ch}:{cx}:{cy},scale={ow}:{fh},setsar=1[cam];\
         [game_src]crop={gw}:{gh}:{gx}:{gy},scale={ow}:{gph},setsar=1[game];\
         [cam][game]vstack=inputs=2",
        cw = cam.w,
        ch = cam.h,
        cx = cam.x,
        cy = cam.y,
        gw = game.w,
        gh = game.h,
        gx = game.x,
        gy = game.y,
        ow = STREAMER_OUT_W,
        fh = STREAMER_FACECAM_H,
        gph = STREAMER_GAMEPLAY_H,
    );
    match drawtext_filters {
        Some(drawtext) if !drawtext.is_empty() => {
            graph.push_str("[stacked];[stacked]");
            graph.push_str(drawtext);
            graph.push_str("[v]");
        }
        _ => graph.push_str("[v]"),
    }
    graph
}

/// Grabs one frame of the source as a JPEG data URL, for the facecam picker.
pub fn preview_frame_data_url(source_path: &str, at_sec: f64) -> Result<String> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }
    let at = format!("{:.3}", at_sec.max(0.0));
    let output = Command::new("ffmpeg")
        .args([
            "-v", "error", "-ss", &at, "-i", source_path, "-frames:v", "1", "-vf",
            "scale='min(960,iw)':-2", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "pipe:1",
        ])
        .output()
        .context("running ffmpeg preview frame")?;

    if !output.status.success() || output.stdout.is_empty() {
        return Err(anyhow!(
            "could not grab a preview frame: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(format!("data:image/jpeg;base64,{}", base64_encode(&output.stdout)))
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Sample rate used for loudness analysis. Plenty for level measurement and cheap to decode.
const LOUDNESS_SAMPLE_RATE: usize = 8000;
/// Quietest level treated as a potential reaction (dBFS). Filters spikes out of near-silence.
const SPIKE_FLOOR_DB: f32 = -35.0;
/// How far above the local baseline a second must be to count as a spike.
const SPIKE_THRESHOLD_DB: f32 = 8.0;
/// Baseline window: the median level over this many seconds either side.
const BASELINE_HALF_WINDOW: usize = 60;

/// Decodes the audio track and returns its RMS level (dBFS) for every second.
/// Streams the PCM from FFmpeg, so a 6-hour VOD never has to sit in memory.
pub fn loudness_per_second(source_path: &str) -> Result<Vec<f32>> {
    use std::io::Read;
    use std::process::Stdio;

    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }
    let rate = LOUDNESS_SAMPLE_RATE.to_string();
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i", source_path, "-vn", "-ac", "1", "-ar", &rate])
        .args(["-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("starting ffmpeg loudness analysis")?;
    let mut stdout = child.stdout.take().context("capturing ffmpeg output")?;

    let mut levels = Vec::new();
    let mut buf = vec![0u8; LOUDNESS_SAMPLE_RATE * 2];
    let (mut sum_sq, mut count, mut carry) = (0f64, 0usize, None::<u8>);
    loop {
        let read = stdout.read(&mut buf).context("reading ffmpeg audio")?;
        if read == 0 {
            break;
        }
        let mut bytes = buf[..read].iter().copied();
        loop {
            let lo = match carry.take().or_else(|| bytes.next()) {
                Some(b) => b,
                None => break,
            };
            let hi = match bytes.next() {
                Some(b) => b,
                None => {
                    carry = Some(lo);
                    break;
                }
            };
            let sample = i16::from_le_bytes([lo, hi]) as f64 / 32768.0;
            sum_sq += sample * sample;
            count += 1;
            if count == LOUDNESS_SAMPLE_RATE {
                levels.push(rms_db(sum_sq, count));
                sum_sq = 0.0;
                count = 0;
            }
        }
    }
    if count > LOUDNESS_SAMPLE_RATE / 4 {
        levels.push(rms_db(sum_sq, count));
    }
    let status = child.wait().context("waiting for ffmpeg")?;
    if !status.success() && levels.is_empty() {
        return Err(anyhow!("ffmpeg could not decode the audio for loudness analysis"));
    }
    Ok(levels)
}

fn rms_db(sum_sq: f64, count: usize) -> f32 {
    let rms = (sum_sq / count.max(1) as f64).sqrt();
    (20.0 * rms.max(1e-6).log10()) as f32
}

/// Finds moments that are much louder than the stream around them.
///
/// Each second is compared with the median of the surrounding two minutes, so a loud
/// game section doesn't drown out reactions, and quiet talk doesn't count as a spike.
/// Neighbouring loud seconds (gaps up to 2 s) merge into one spike.
pub fn find_loudness_spikes(levels: &[f32]) -> Vec<LoudnessSpike> {
    let n = levels.len();
    let mut spikes: Vec<LoudnessSpike> = Vec::new();
    let mut window: Vec<f32> = Vec::with_capacity(BASELINE_HALF_WINDOW * 2 + 1);

    for (i, &level) in levels.iter().enumerate() {
        if level < SPIKE_FLOOR_DB {
            continue;
        }
        let lo = i.saturating_sub(BASELINE_HALF_WINDOW);
        let hi = (i + BASELINE_HALF_WINDOW + 1).min(n);
        window.clear();
        window.extend_from_slice(&levels[lo..hi]);
        window.sort_by(|a, b| a.total_cmp(b));
        let baseline = window[window.len() / 2];
        let excess = level - baseline;
        if excess < SPIKE_THRESHOLD_DB {
            continue;
        }

        let t = i as f64;
        match spikes.last_mut() {
            Some(last) if t - last.end <= 2.0 => {
                last.end = t + 1.0;
                if excess as f64 > last.excess_db {
                    last.excess_db = excess as f64;
                    last.peak_db = level as f64;
                    last.peak_sec = t;
                }
            }
            _ => spikes.push(LoudnessSpike {
                start: t,
                end: t + 1.0,
                peak_sec: t,
                peak_db: level as f64,
                excess_db: excess as f64,
            }),
        }
    }
    spikes
}

/// One piece of a long recording that is transcribed on its own.
///
/// Pieces overlap so no word is cut in half at a boundary. Each piece only *keeps*
/// the words whose midpoint falls in `keep_from..keep_to`, so overlapping words are
/// counted exactly once when the pieces are stitched back together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioChunk {
    pub index: usize,
    /// Where extraction starts in the source, in seconds.
    pub start: f64,
    /// Where extraction ends; `None` means "to the end of the file".
    pub end: Option<f64>,
    pub keep_from: f64,
    pub keep_to: f64,
}

/// Splits a recording of `duration` seconds into `chunk_secs` pieces with `overlap_secs`
/// of extra audio on each side. A recording that fits in one piece stays whole, and a
/// short leftover at the end is folded into the previous piece.
pub fn plan_audio_chunks(duration: f64, chunk_secs: f64, overlap_secs: f64) -> Vec<AudioChunk> {
    let whole = AudioChunk { index: 0, start: 0.0, end: None, keep_from: 0.0, keep_to: f64::MAX };
    if !duration.is_finite() || duration <= chunk_secs + overlap_secs {
        return vec![whole];
    }

    let mut count = (duration / chunk_secs).ceil() as usize;
    // Avoid a tiny final piece (under a quarter of a chunk): merge it into the one before.
    if duration - (count - 1) as f64 * chunk_secs < chunk_secs / 4.0 {
        count -= 1;
    }

    (0..count)
        .map(|index| {
            let last = index + 1 == count;
            let keep_from = index as f64 * chunk_secs;
            let keep_to = if last { f64::MAX } else { (index + 1) as f64 * chunk_secs };
            AudioChunk {
                index,
                start: (keep_from - overlap_secs).max(0.0),
                // The last piece runs to the end of the file, in case the probed length is short.
                end: if last { None } else { Some(keep_to + overlap_secs) },
                keep_from,
                keep_to,
            }
        })
        .collect()
}

/// Cuts one piece of audio out of the source as 16 kHz mono. The format follows the
/// output extension (`.flac` for upload, `.wav` for local Whisper).
pub fn extract_audio_chunk(source_path: &str, chunk: &AudioChunk, output_path: &Path) -> Result<()> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let start = format!("{:.3}", chunk.start);
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-ss", &start, "-i", source_path]);
    if let Some(end) = chunk.end {
        cmd.args(["-t", &format!("{:.3}", (end - chunk.start).max(0.1))]);
    }
    cmd.args(["-vn", "-ac", "1", "-ar", "16000"]).arg(output_path);

    let output = cmd.output().context("running ffmpeg audio chunk extraction")?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg audio extraction failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> NormRect {
        NormRect { x, y, w, h }
    }

    #[test]
    fn gameplay_crop_is_centered_27_32_on_1080p() {
        let c = fit_crop_to_aspect(1920, 1080, &rect(0.0, 0.0, 1.0, 1.0), 1080, 1280);
        assert_eq!(c.h, 1080);
        assert_eq!(c.w, 910); // 1080 * 27/32 = 911.25 -> even floor
        assert_eq!(c.x, (1920 - 910) / 2);
        assert_eq!(c.y, 0);
    }

    #[test]
    fn facecam_crop_matches_band_aspect_and_stays_in_frame() {
        // Bottom-right webcam, slightly off-aspect selection.
        let c = fit_crop_to_aspect(1920, 1080, &rect(0.75, 0.70, 0.25, 0.30), 1080, 640);
        assert!(c.x + c.w <= 1920 && c.y + c.h <= 1080);
        assert_eq!(c.w % 2, 0);
        assert_eq!(c.h % 2, 0);
        let aspect = c.w as f64 / c.h as f64;
        assert!((aspect - 1080.0 / 640.0).abs() < 0.02, "aspect {aspect}");
    }

    #[test]
    fn out_of_range_or_degenerate_rects_do_not_panic() {
        for r in [
            rect(1.5, -1.0, 2.0, 3.0),
            rect(0.5, 0.5, 0.0, 0.0),
            rect(f64::NAN, 0.0, 0.2, 0.2),
            rect(0.99, 0.99, 0.5, 0.5),
        ] {
            let c = fit_crop_to_aspect(1280, 720, &r, 1080, 640);
            assert!(c.w >= 2 && c.h >= 2);
            assert!(c.x >= 0 && c.y >= 0 && c.x + c.w <= 1280 && c.y + c.h <= 720, "{c:?}");
        }
    }

    #[test]
    fn filter_graph_labels_output_with_and_without_captions() {
        let cam = rect(0.0, 0.0, 0.3, 0.3);
        let plain = streamer_filter_graph(1920, 1080, &cam, None);
        assert!(plain.ends_with("vstack=inputs=2[v]"));
        let captioned = streamer_filter_graph(1920, 1080, &cam, Some("drawtext=text='HI'"));
        assert!(captioned.ends_with("[stacked];[stacked]drawtext=text='HI'[v]"));
    }

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn loudness_spikes_are_relative_to_local_baseline() {
        // 10 minutes of steady talk at -30 dB, a scream at 5:00, and a loud but steady
        // game section from 7:00 onwards that should NOT count as a spike.
        let mut levels = vec![-30.0f32; 600];
        for l in levels.iter_mut().skip(420) {
            *l = -14.0;
        }
        levels[300] = -12.0;
        levels[301] = -10.0;
        levels[303] = -13.0; // within the 2 s gap, merges with the first two
        let spikes = find_loudness_spikes(&levels);
        assert_eq!(spikes.len(), 1, "{spikes:?}");
        let s = spikes[0];
        assert_eq!((s.start, s.end, s.peak_sec), (300.0, 304.0, 301.0));
        assert!((s.excess_db - 20.0).abs() < 0.01);
    }

    #[test]
    fn quiet_streams_produce_no_spikes() {
        let mut levels = vec![-60.0f32; 300];
        levels[100] = -45.0; // 15 dB jump, but still below the floor
        assert!(find_loudness_spikes(&levels).is_empty());
    }

    #[test]
    fn loudness_of_a_real_file_is_measured_per_second() {
        if !command_exists("ffmpeg") {
            return;
        }
        let dir = std::env::temp_dir().join(format!("autoshorts-loud-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("tone.wav");
        // 3 s quiet tone, 1 s loud tone, 3 s quiet tone.
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("sine=frequency=300:duration=7,volume='if(between(t,3,4),1,0.05)':eval=frame")
            .arg(&wav)
            .status()
            .unwrap();
        assert!(status.success());
        let levels = loudness_per_second(wav.to_str().unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(levels.len(), 7, "{levels:?}");
        assert!(levels[3] - levels[1] > 20.0, "{levels:?}");
    }

    #[test]
    fn short_recordings_stay_in_one_piece() {
        let chunks = plan_audio_chunks(1250.0, 1200.0, 10.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end, None);
    }

    #[test]
    fn six_hour_stream_is_covered_by_overlapping_pieces() {
        let duration = 6.0 * 3600.0 + 125.0;
        let chunks = plan_audio_chunks(duration, 1200.0, 10.0);
        // 6 h 2 min: 18 full pieces; the 2-minute leftover joins the last one.
        assert_eq!(chunks.len(), 18);
        assert_eq!(chunks[0].start, 0.0);
        assert_eq!(chunks[0].end, Some(1210.0));
        assert_eq!(chunks[1].start, 1190.0);
        assert_eq!(chunks[17].end, None);
        // Keep ranges tile the timeline with no gaps and no double counting.
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].keep_to, pair[1].keep_from);
        }
        assert_eq!(chunks[17].keep_to, f64::MAX);
    }

    #[test]
    fn audio_chunk_is_cut_with_requested_length() {
        if !command_exists("ffmpeg") {
            return;
        }
        let dir = std::env::temp_dir().join(format!("autoshorts-chunk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.wav");
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=30"])
            .arg(&src)
            .status()
            .unwrap()
            .success();
        assert!(ok);
        let chunk = AudioChunk { index: 1, start: 10.0, end: Some(22.0), keep_from: 12.0, keep_to: 20.0 };
        let out = dir.join("part.flac");
        extract_audio_chunk(src.to_str().unwrap(), &chunk, &out).unwrap();
        let probe = probe_media(out.to_str().unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let d = probe.duration_sec.unwrap();
        assert!((d - 12.0).abs() < 0.1, "{d}");
    }
}
