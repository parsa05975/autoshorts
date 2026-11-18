use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::models::{CandidateDraft, LoudnessSpike, NormalizedTranscript, TranscriptSegment};

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeepseekMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeepseekChoice {
    message: DeepseekMessage,
}

#[derive(Debug, Deserialize)]
struct DeepseekResponse {
    choices: Vec<DeepseekChoice>,
}

/// DeepSeek model to call. `deepseek-chat` and `deepseek-reasoner` were retired on
/// 2026-07-24, so those (and an empty setting) map to the current `deepseek-flash`.
fn deepseek_model(requested: Option<String>) -> String {
    match requested.as_deref().map(str::trim) {
        None | Some("") | Some("deepseek-chat") | Some("deepseek-reasoner") => "deepseek-flash".to_string(),
        Some(name) => name.to_string(),
    }
}

async fn ask_deepseek(
    prompt: &MomentPrompt,
    api_key: &str,
    model_name: Option<&str>,
) -> Result<String> {
    let prompt = prompt.combined();

    let model = deepseek_model(
        model_name
            .map(str::to_string)
            .or_else(|| std::env::var("DEEPSEEK_MODEL").ok()),
    );

    let response = reqwest::Client::new()
        .post("https://api.deepseek.com/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            // Picking clips doesn't need step-by-step reasoning; thinking mode would
            // only make each part slower and several times more expensive.
            "thinking": {
                "type": "disabled"
            },
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling DeepSeek")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("DeepSeek request failed ({status}): {body}"));
    }

    let res_body: DeepseekResponse = response.json().await.context("parsing DeepSeek response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("DeepSeek response did not include choices content"))?;

    Ok(text)
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}

async fn ask_gemini(
    prompt: &MomentPrompt,
    api_key: &str,
) -> Result<String> {
    let prompt = prompt.combined();

    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let response = reqwest::Client::new()
        .post(&url)
        .json(&json!({
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }))
        .send()
        .await
        .context("calling Gemini")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Gemini request failed ({status}): {body}"));
    }

    let res_body: GeminiResponse = response.json().await.context("parsing Gemini response")?;
    let text = res_body
        .candidates
        .first()
        .and_then(|c| c.content.parts.first())
        .and_then(|p| p.text.clone())
        .ok_or_else(|| anyhow!("Gemini response did not include content text"))?;

    Ok(text)
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: String,
}

async fn ask_openai(
    prompt: &MomentPrompt,
    api_key: &str,
) -> Result<String> {
    let prompt = prompt.combined();

    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let response = reqwest::Client::new()
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling OpenAI")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("OpenAI request failed ({status}): {body}"));
    }

    let res_body: ChatCompletionResponse =
        response.json().await.context("parsing OpenAI response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("OpenAI response did not include choices content"))?;

    Ok(text)
}

async fn ask_openrouter(
    prompt: &MomentPrompt,
    api_key: &str,
    model_name: Option<&str>,
) -> Result<String> {
    let prompt = prompt.combined();

    let default_model = "google/gemini-2.5-flash".to_string();
    let model = model_name
        .filter(|m| !m.trim().is_empty())
        .map(|m| m.trim().to_string())
        .or_else(|| std::env::var("OPENROUTER_MODEL").ok().filter(|m| !m.trim().is_empty()))
        .unwrap_or(default_model);

    let response = reqwest::Client::new()
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling OpenRouter")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("OpenRouter request failed ({status}): {body}"));
    }

    let res_body: ChatCompletionResponse = response
        .json()
        .await
        .context("parsing OpenRouter response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("OpenRouter response did not include choices content"))?;

    Ok(text)
}

async fn ask_groq(
    prompt: &MomentPrompt,
    api_key: &str,
) -> Result<String> {
    let prompt = prompt.combined();

    let model =
        std::env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.3-70b-versatile".to_string());

    let response = reqwest::Client::new()
        .post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling Groq")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Groq request failed ({status}): {body}"));
    }

    let res_body: ChatCompletionResponse = response
        .json()
        .await
        .context("parsing Groq response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("Groq response did not include choices content"))?;

    Ok(text)
}

#[derive(Debug, Serialize)]
struct ClaudeMessage<'a> {
    role: &'a str,
    content: String,
}

async fn ask_claude(
    prompt: &MomentPrompt,
    api_key: &str,
) -> Result<String> {
    let prompt = prompt.combined();

    let model =
        std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".to_string());

    let response = reqwest::Client::new()
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": model,
            "max_tokens": 4096,
            "temperature": 0.2,
            "messages": [
                ClaudeMessage {
                    role: "user",
                    content: prompt,
                }
            ]
        }))
        .send()
        .await
        .context("calling Claude")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Claude request failed ({status}): {body}"));
    }

    let message: AnthropicMessage = response.json().await.context("parsing Claude response")?;
    let text = message
        .content
        .into_iter()
        .find_map(|content| content.text)
        .ok_or_else(|| anyhow!("Claude response did not include text content"))?;

    Ok(text)
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
}

async fn ask_local_llm(
    prompt: &MomentPrompt,
    model_name: &str,
) -> Result<String> {
    let system_instructions = format!(
        "{MOMENT_INSTRUCTIONS}\n\n\
REMINDERS FOR THIS MODEL: Each clip MUST last between 20 and 60 seconds (end minus start). \
Combine adjacent transcript lines to reach that length. Return at least 3 candidates; never an empty list. \
start and end must be real timestamps from the transcript, never 0.0 for both."
    );
    let user_content = prompt.input.clone();

    let response = reqwest::Client::new()
        .post("http://localhost:11434/api/chat")
        .json(&json!({
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_instructions,
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_ctx": 16384
            },
            "format": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": { "type": "number" },
                                "end": { "type": "number" },
                                "score": { "type": "number" },
                                "hook": { "type": "string" },
                                "rationale": { "type": "string" }
                            },
                            "required": ["start", "end", "score", "hook", "rationale"]
                        }
                    }
                },
                "required": ["candidates"]
            }
        }))
        .send()
        .await
        .context("calling local Ollama")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Local Ollama request failed ({status}): {body}"));
    }

    let res_body: OllamaResponse = response
        .json()
        .await
        .context("parsing local Ollama response")?;
    Ok(res_body.message.content)
}

/// Instructions shared by every provider: what a great GTA 6 stream clip is and how to cut it.
const MOMENT_INSTRUCTIONS: &str = "You are the clip editor for a GTA 6 livestream channel. \
Find the moments in this stream that would go viral as TikTok / YouTube Shorts gaming clips.

WHAT MAKES A GREAT CLIP, in priority order:
1. Big streamer reactions: screaming, yelling, losing it laughing, stunned silence then an outburst, rage.
2. Police chases and wanted-level chaos: escapes, close calls, getting busted at the last second.
3. Crashes and wipeouts: spectacular car, bike, boat or plane crashes, ragdoll physics.
4. Fails: botched heists or missions, dumb deaths, plans going wrong, falling off things.
5. Funny NPC moments: NPCs saying or doing something absurd, strange AI behaviour, glitches.
6. First-look reveals: the streamer seeing a new place, vehicle, character, mechanic or story beat for the first time and reacting to it.
Skip: menus, loading screens, uneventful driving, reading chat, sponsor reads, tech problems, long calm talk.

USING THE AUDIO:
The best moments are often reactions rather than words, and speech-to-text misses screams and laughter. You get two audio signals:
- LOUDNESS SPIKES: seconds where the audio is far louder than the surrounding two minutes. Usually the streamer screaming or laughing, or a crash, explosion or gunfire in the game. More dB means a bigger moment.
- Transcript lines that overlap a spike are tagged <LOUD +NdB>.
Treat a spike as strong evidence of a moment, especially together with exclamations in the transcript \
(\"no no no\", \"oh my god\", \"bro\", \"what\", \"let's go\", swearing, laughter, repeated words). \
A spike with little or no transcript around it is often a pure reaction or game event and can still be a clip. \
Not every spike is a moment (music, an alert sound, one gunshot in routine play): use the transcript to judge.

HOW TO CUT:
- 20 to 60 seconds per clip.
- Start 5 to 15 seconds BEFORE the peak so viewers see the setup (the chase starting, the car speeding up, the NPC walking over). End shortly after the reaction dies down, not mid-sentence.
- Clips must not overlap each other.
- start and end are seconds and must be real times inside the stream, taken from the timestamps below.

FOR EACH CLIP RETURN:
- start, end: seconds.
- score: 0.0 to 1.0, how likely the clip is to go viral. Reward reaction intensity, a clear payoff and spike size. Be honest: most clips are not 0.9+.
- hook: a short on-screen title in TikTok style, at most 60 characters, that stops the scroll \
(for example \"He did NOT see that cop coming\" or \"GTA 6 NPCs are unhinged\"). No emojis, no hashtags.
- rationale: start with one category tag: [REACTION], [POLICE CHASE], [CRASH], [FAIL], [NPC], [FIRST LOOK] or [OTHER]; \
then one sentence on what happens and why it works.

Return up to 15 clips, best first, as JSON in exactly this shape:
{\"candidates\":[{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}]}";

/// How many spikes are listed in the prompt (the strongest ones).
const MAX_PROMPT_SPIKES: usize = 60;
/// Largest score bonus a clip gets for containing a big loudness spike.
const LOUDNESS_BONUS: f64 = 0.1;
/// Spike size (dB above baseline) that earns the full bonus.
const FULL_BONUS_EXCESS_DB: f64 = 20.0;
/// Shortest clip kept from a long stream.
const MIN_CLIP_SECONDS: f64 = 15.0;
/// Most clips kept for a stream of up to an hour (longer streams keep more).
const MAX_CANDIDATES: usize = 10;

/// Long transcripts are analyzed in parts of this length...
const ANALYSIS_WINDOW_SECS: f64 = 30.0 * 60.0;
/// ...each also seeing this much of the previous part, so a moment on a boundary is seen whole.
const ANALYSIS_OVERLAP_SECS: f64 = 2.0 * 60.0;
/// Parts sent to the AI at once, and tries per part.
const ANALYSIS_CONCURRENCY: usize = 3;
const ANALYSIS_ATTEMPTS: usize = 3;
/// Clips may start or end this far outside their part; anything further is a made-up time.
const WINDOW_SLACK_SECS: f64 = 30.0;

/// Which AI to ask, with what it needs to answer.
#[derive(Debug, Clone)]
pub enum LlmChoice {
    Claude { api_key: String },
    Local { model: String },
    Gemini { api_key: String },
    OpenAi { api_key: String },
    OpenRouter { api_key: String, model: Option<String> },
    Groq { api_key: String },
    DeepSeek { api_key: String, model: Option<String> },
}

impl LlmChoice {
    async fn ask(&self, prompt: &MomentPrompt) -> Result<String> {
        match self {
            Self::Claude { api_key } => ask_claude(prompt, api_key).await,
            Self::Local { model } => ask_local_llm(prompt, model).await,
            Self::Gemini { api_key } => ask_gemini(prompt, api_key).await,
            Self::OpenAi { api_key } => ask_openai(prompt, api_key).await,
            Self::OpenRouter { api_key, model } => ask_openrouter(prompt, api_key, model.as_deref()).await,
            Self::Groq { api_key } => ask_groq(prompt, api_key).await,
            Self::DeepSeek { api_key, model } => ask_deepseek(prompt, api_key, model.as_deref()).await,
        }
    }

    /// Local models share one machine; cloud APIs can take a few parts at once.
    fn concurrency(&self) -> usize {
        match self {
            Self::Local { .. } => 1,
            _ => ANALYSIS_CONCURRENCY,
        }
    }
}

/// The two halves of a request: fixed instructions and this part's data.
#[derive(Debug, Clone)]
struct MomentPrompt {
    input: String,
}

impl MomentPrompt {
    /// Instructions and data as one message, for providers that take a single prompt.
    fn combined(&self) -> String {
        format!("{MOMENT_INSTRUCTIONS}\n\n{}", self.input)
    }
}

/// One slice of the stream that is analyzed on its own.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AnalysisWindow {
    index: usize,
    start: f64,
    end: f64,
}

/// Splits a stream into parts. Each part starts a little before the previous one ends.
/// A short stream is one part; a short leftover is folded into the last part.
fn plan_windows(duration: f64) -> Vec<AnalysisWindow> {
    if duration <= ANALYSIS_WINDOW_SECS + ANALYSIS_OVERLAP_SECS {
        return vec![AnalysisWindow { index: 0, start: 0.0, end: duration }];
    }
    let mut count = (duration / ANALYSIS_WINDOW_SECS).ceil() as usize;
    if duration - (count - 1) as f64 * ANALYSIS_WINDOW_SECS < ANALYSIS_WINDOW_SECS / 4.0 {
        count -= 1;
    }
    (0..count)
        .map(|index| AnalysisWindow {
            index,
            start: (index as f64 * ANALYSIS_WINDOW_SECS - ANALYSIS_OVERLAP_SECS).max(0.0),
            end: if index + 1 == count { duration } else { (index + 1) as f64 * ANALYSIS_WINDOW_SECS },
        })
        .collect()
}

/// Most clips kept: 10 for streams up to an hour, 3 more per extra hour, at most 25.
fn max_candidates(duration: f64) -> usize {
    let extra_hours = ((duration - 3600.0) / 3600.0).ceil().max(0.0) as usize;
    (MAX_CANDIDATES + 3 * extra_hours).min(25)
}

/// Finds clip candidates in a transcript of any length.
///
/// The stream is analyzed part by part (see [`plan_windows`]); every part is retried
/// before giving up, so no stretch of the stream is silently skipped. `progress` is
/// called with (parts done, total parts).
pub async fn detect_candidates(
    llm: &LlmChoice,
    transcript: &NormalizedTranscript,
    spikes: &[LoudnessSpike],
    progress: impl Fn(usize, usize),
) -> Result<Vec<CandidateDraft>> {
    let duration = transcript
        .duration
        .max(transcript.segments.last().map(|s| s.end).unwrap_or(0.0));
    let windows = plan_windows(duration);
    let total = windows.len();
    let min_duration = min_clip_duration(transcript);

    progress(0, total);
    let mut pending = windows.into_iter();
    let mut running = tokio::task::JoinSet::new();
    let mut found = Vec::new();
    let mut done = 0;
    loop {
        while running.len() < llm.concurrency() {
            let Some(window) = pending.next() else { break };
            let prompt = MomentPrompt { input: moment_input(transcript, spikes, duration, window, total) };
            let llm = llm.clone();
            running.spawn(async move {
                let mut last_error = anyhow!("no attempt made");
                for attempt in 0..ANALYSIS_ATTEMPTS {
                    if attempt > 0 {
                        tokio::time::sleep(std::time::Duration::from_secs(5 * attempt as u64)).await;
                    }
                    match llm.ask(&prompt).await.and_then(|text| parse_candidate_json(&text, min_duration)) {
                        Ok(candidates) => return (window, Ok(candidates)),
                        // A missing local model won't appear by retrying; report it straight away.
                        Err(error) if error.to_string().to_lowercase().contains("not found") => {
                            return (window, Err(error))
                        }
                        Err(error) => last_error = error,
                    }
                }
                (window, Err(last_error))
            });
        }
        let Some(joined) = running.join_next().await else { break };
        let (window, result) = joined.context("analysis task failed")?;
        match result {
            Ok(candidates) => {
                found.extend(candidates.into_iter().filter(|c| {
                    c.end > c.start
                        && c.start >= window.start - WINDOW_SLACK_SECS
                        && c.end <= window.end + WINDOW_SLACK_SECS
                }));
                done += 1;
                progress(done, total);
            }
            Err(error) if total == 1 => return Err(error),
            Err(error) => {
                return Err(error.context(format!(
                    "analysis stopped at part {} of {} ({}-{})",
                    window.index + 1,
                    total,
                    format_clock(window.start),
                    format_clock(window.end)
                )))
            }
        }
    }
    Ok(rank_candidates(found, spikes, max_candidates(duration)))
}

pub(crate) fn format_clock(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// The data part of a request: where this part sits in the stream, its loudest
/// moments, and its transcript with loud lines tagged.
fn moment_input(
    transcript: &NormalizedTranscript,
    spikes: &[LoudnessSpike],
    stream_duration: f64,
    window: AnalysisWindow,
    total_windows: usize,
) -> String {
    let in_window = |start: f64, end: f64| start < window.end && end > window.start;
    let part_spikes: Vec<LoudnessSpike> = spikes.iter().copied().filter(|s| in_window(s.start, s.end)).collect();
    let part_segments: Vec<TranscriptSegment> = transcript
        .segments
        .iter()
        .filter(|s| in_window(s.start, s.end))
        .cloned()
        .collect();

    let mut strongest: Vec<&LoudnessSpike> = part_spikes.iter().collect();
    strongest.sort_by(|a, b| b.excess_db.total_cmp(&a.excess_db));
    strongest.truncate(MAX_PROMPT_SPIKES);
    strongest.sort_by(|a, b| a.start.total_cmp(&b.start));

    let spike_lines = if strongest.is_empty() {
        "None detected in this part (no audio track, or the audio level stays steady).".to_string()
    } else {
        strongest
            .iter()
            .map(|s| {
                format!(
                    "- {:.1}-{:.1} s (peak at {:.1} s): +{:.1} dB",
                    s.start, s.end, s.peak_sec, s.excess_db
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let part_line = if total_windows > 1 {
        format!(
            "THIS IS PART {} OF {}: {} to {} ({:.0}-{:.0} s). Only pick clips inside this part. \
It is fine to return few or no clips if nothing good happens here.\n",
            window.index + 1,
            total_windows,
            format_clock(window.start),
            format_clock(window.end),
            window.start,
            window.end
        )
    } else {
        String::new()
    };

    format!(
        "STREAM LENGTH: {} ({:.0} seconds)\n{part_line}\n\
LOUDNESS SPIKES (strongest {} shown, in time order; dB above the local baseline):\n{}\n\n\
TRANSCRIPT (timestamps in seconds):\n{}",
        format_clock(stream_duration),
        stream_duration,
        strongest.len(),
        spike_lines,
        compact_segments(&part_segments, &part_spikes)
    )
}

/// Biggest spike (dB above baseline) that overlaps the given time range.
fn loudest_overlap(spikes: &[LoudnessSpike], start: f64, end: f64) -> Option<f64> {
    spikes
        .iter()
        .filter(|s| s.start < end && s.end > start)
        .map(|s| s.excess_db)
        .max_by(|a, b| a.total_cmp(b))
}

fn min_clip_duration(transcript: &NormalizedTranscript) -> f64 {
    if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        MIN_CLIP_SECONDS
    }
}

/// Boosts clips containing big loudness spikes, drops clips that mostly repeat a
/// better one (parts overlap, so the same moment can come back twice), keeps the best.
fn rank_candidates(
    mut candidates: Vec<CandidateDraft>,
    spikes: &[LoudnessSpike],
    limit: usize,
) -> Vec<CandidateDraft> {
    for c in candidates.iter_mut() {
        if let Some(excess) = loudest_overlap(spikes, c.start, c.end) {
            c.score = (c.score + LOUDNESS_BONUS * (excess / FULL_BONUS_EXCESS_DB).min(1.0)).min(1.0);
        }
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut kept: Vec<CandidateDraft> = Vec::new();
    for c in candidates {
        let duplicate = kept.iter().any(|k| {
            let overlap = (c.end.min(k.end) - c.start.max(k.start)).max(0.0);
            overlap > 0.5 * (c.end - c.start).min(k.end - k.start)
        });
        if !duplicate {
            kept.push(c);
        }
        if kept.len() == limit {
            break;
        }
    }
    kept
}

fn compact_segments(segments: &[TranscriptSegment], spikes: &[LoudnessSpike]) -> String {
    segments
        .iter()
        .map(|segment| {
            let speaker = segment.speaker.as_deref().unwrap_or("Speaker");
            let loud = loudest_overlap(spikes, segment.start, segment.end)
                .map(|db| format!(" <LOUD +{db:.0}dB>"))
                .unwrap_or_default();
            format!(
                "[{:.2}-{:.2}] {}: {}{}",
                segment.start, segment.end, speaker, segment.text, loud
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_candidate_json(text: &str, min_duration: f64) -> Result<Vec<CandidateDraft>> {
    let trimmed = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let val: serde_json::Value = serde_json::from_str(trimmed).context("parsing candidate JSON")?;

    let candidates_arr = if val.is_array() {
        val.as_array().cloned()
    } else if val.is_object() {
        let mut found_arr = None;
        for key in &[
            "candidates",
            "Candidates",
            "moments",
            "clips",
            "segments",
            "results",
        ] {
            if let Some(arr) = val.get(*key).and_then(|v| v.as_array()) {
                found_arr = Some(arr.clone());
                break;
            }
        }
        if found_arr.is_none() {
            if let Some(obj) = val.as_object() {
                for (_key, value) in obj {
                    if let Some(arr) = value.as_array() {
                        found_arr = Some(arr.clone());
                        break;
                    }
                }
            }
        }
        if found_arr.is_some() {
            found_arr
        } else if val.get("start").is_some() && val.get("end").is_some() {
            Some(vec![val.clone()])
        } else {
            None
        }
    } else {
        None
    };

    let concrete_arr = candidates_arr.ok_or_else(|| {
        anyhow!(
            "Ollama output does not contain a candidates array. Raw output: {}",
            trimmed
        )
    })?;

    let mut drafts = Vec::new();
    for item in &concrete_arr {
        let start = match item.get("start") {
            Some(v) => {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(s) = v.as_str() {
                    s.parse::<f64>().unwrap_or(0.0)
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    0.0
                }
            }
            None => 0.0,
        };

        let end = match item.get("end") {
            Some(v) => {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(s) = v.as_str() {
                    s.parse::<f64>().unwrap_or(0.0)
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    0.0
                }
            }
            None => 0.0,
        };

        let mut score = match item.get("score") {
            Some(v) => {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(s) = v.as_str() {
                    s.parse::<f64>().unwrap_or(0.8)
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    0.8
                }
            }
            None => 0.8,
        };

        if score > 1.0 && score <= 10.0 {
            score /= 10.0;
        } else if score > 10.0 && score <= 100.0 {
            score /= 100.0;
        } else if score > 100.0 {
            score = 1.0;
        } else if score < 0.0 {
            score = 0.0;
        }

        let hook = item
            .get("hook")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let rationale = item
            .get("rationale")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        drafts.push(CandidateDraft {
            start,
            end,
            score,
            hook,
            rationale,
        });
    }

    let mut candidates = drafts
        .clone()
        .into_iter()
        .filter(|candidate| {
            (candidate.end - candidate.start) >= min_duration && !candidate.hook.trim().is_empty()
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        candidates = drafts
            .into_iter()
            .filter(|candidate| {
                (candidate.end - candidate.start) >= 5.0 && !candidate.hook.trim().is_empty()
            })
            .collect::<Vec<_>>();
    }

    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(duration: f64) -> NormalizedTranscript {
        NormalizedTranscript {
            language: "en".into(),
            duration,
            speakers: vec![],
            words: vec![],
            segments: vec![
                TranscriptSegment { start: 100.0, end: 104.0, speaker: None, text: "chilling".into() },
                TranscriptSegment { start: 200.0, end: 203.5, speaker: None, text: "NO NO NO".into() },
                TranscriptSegment { start: 5000.0, end: 5004.0, speaker: None, text: "later on".into() },
            ],
        }
    }

    fn spike(start: f64, end: f64, excess_db: f64) -> LoudnessSpike {
        LoudnessSpike { start, end, peak_sec: start, peak_db: -10.0, excess_db }
    }

    fn draft(start: f64, end: f64, score: f64, hook: &str) -> CandidateDraft {
        CandidateDraft { start, end, score, hook: hook.into(), rationale: "[OTHER] x".into() }
    }

    #[test]
    fn prompt_lists_spikes_and_tags_loud_lines() {
        let t = transcript(7300.0);
        let window = AnalysisWindow { index: 0, start: 0.0, end: 1800.0 };
        let input = moment_input(&t, &[spike(201.0, 203.0, 17.2), spike(5001.0, 5002.0, 9.0)], 7300.0, window, 5);
        assert!(input.contains("STREAM LENGTH: 2:01:40"));
        assert!(input.contains("THIS IS PART 1 OF 5: 0:00:00 to 0:30:00"));
        assert!(input.contains("- 201.0-203.0 s (peak at 201.0 s): +17.2 dB"));
        assert!(input.contains("[200.00-203.50] Speaker: NO NO NO <LOUD +17dB>"));
        // Other parts' lines and spikes stay out of this part's prompt.
        assert!(!input.contains("later on") && !input.contains("5001.0"));
        let prompt = MomentPrompt { input }.combined();
        assert!(prompt.starts_with("You are the clip editor for a GTA 6"));
    }

    #[test]
    fn six_hour_stream_is_analyzed_in_overlapping_parts() {
        let windows = plan_windows(6.0 * 3600.0);
        assert_eq!(windows.len(), 12);
        assert_eq!((windows[0].start, windows[0].end), (0.0, 1800.0));
        assert_eq!((windows[1].start, windows[1].end), (1680.0, 3600.0));
        assert_eq!(windows[11].end, 21600.0);
        for pair in windows.windows(2) {
            assert!(pair[1].start < pair[0].end, "gap between parts");
        }
        assert_eq!(plan_windows(1500.0).len(), 1);
        assert_eq!(max_candidates(1500.0), 10);
        assert_eq!(max_candidates(6.0 * 3600.0), 25);
    }

    #[test]
    fn loud_clips_are_boosted_and_overlapping_clips_dropped() {
        let ranked = rank_candidates(
            vec![
                draft(180.0, 220.0, 0.80, "chase"),
                draft(185.0, 225.0, 0.70, "same chase"),
                draft(400.0, 440.0, 0.85, "quiet"),
            ],
            &[spike(201.0, 203.0, 20.0)],
            10,
        );
        let hooks: Vec<&str> = ranked.iter().map(|c| c.hook.as_str()).collect();
        assert_eq!(hooks, ["chase", "quiet"]);
        assert!((ranked[0].score - 0.90).abs() < 1e-9);
    }

    #[test]
    fn retired_deepseek_names_map_to_the_current_model() {
        assert_eq!(deepseek_model(None), "deepseek-flash");
        assert_eq!(deepseek_model(Some("  ".into())), "deepseek-flash");
        assert_eq!(deepseek_model(Some("deepseek-chat".into())), "deepseek-flash");
        assert_eq!(deepseek_model(Some("deepseek-v4-pro".into())), "deepseek-v4-pro");
    }

    #[test]
    fn short_clips_are_rejected_when_parsing() {
        let answer = r#"{"candidates":[
            {"start":180,"end":220,"score":0.8,"hook":"ok","rationale":"[CRASH] x"},
            {"start":500,"end":505,"score":0.99,"hook":"too short","rationale":"[FAIL] x"}
        ]}"#;
        let parsed = parse_candidate_json(answer, MIN_CLIP_SECONDS).unwrap();
        assert_eq!(parsed.len(), 1);
    }
}
