use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::models::{ClipLayout, MediaProbe, NormRect};

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

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-ss", &start, "-i", source_path, "-t", &duration]);

    let streamer_facecam = match layout {
        ClipLayout::Streamer { facecam } if has_video => Some(*facecam),
        _ => None,
    };

    if let Some(facecam) = streamer_facecam {
        let (src_w, src_h) = match probe.as_ref().and_then(|p| p.width.zip(p.height)) {
            Some((w, h)) if w > 0 && h > 0 => (w, h),
            _ => {
                return Err(anyhow!(
                    "streamer layout needs the video's size, but ffprobe could not read it"
                ))
            }
        };
        let graph = streamer_filter_graph(src_w, src_h, &facecam, drawtext_filters);
        cmd.args(["-filter_complex", &graph, "-map", "[v]", "-map", "0:a?"]);
        cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]);
    } else if has_video {
        let mut filter = "crop=w='2*trunc(min(iw,ih*9/16)/2)':h='2*trunc(min(ih,iw*16/9)/2)'".to_string();
        if let Some(drawtext) = drawtext_filters {
            if !drawtext.is_empty() {
                filter = format!("{},{}", filter, drawtext);
            }
        }
        cmd.args(["-vf", &filter]);
        cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]);
    } else {
        cmd.arg("-vn");
    }

    cmd.args(["-c:a", "aac", "-b:a", "192k"]);
    cmd.arg(output_path);

    let output = cmd.output().context("running ffmpeg clip render")?;

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
}
