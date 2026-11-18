import React, { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Check, Crop, Gamepad2, Loader2, RefreshCw, Video } from "lucide-react";

export type NormRect = { x: number; y: number; w: number; h: number };
export type ClipLayout = { mode: "center" } | { mode: "streamer"; facecam: NormRect };

// Must match STREAMER_OUT_W / STREAMER_FACECAM_H / STREAMER_GAMEPLAY_H in src-tauri/src/media.rs.
const OUT_W = 1080;
const FACECAM_H = 640;
const GAMEPLAY_H = 1280;
const BAND_ASPECT = OUT_W / FACECAM_H;
const GAMEPLAY_ASPECT = OUT_W / GAMEPLAY_H;

const LAST_FACECAM_KEY = "autoshorts_last_facecam";
const MIN_W = 0.05;

type Corner = "tl" | "tr" | "bl" | "br";
type DragState =
  | { kind: "move"; startX: number; startY: number; origin: NormRect }
  | { kind: "resize"; corner: Corner; origin: NormRect };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Height (as a fraction of the frame) that keeps a box of width `w` at the facecam band's shape. */
function heightFor(w: number, srcAspect: number) {
  return (w * srcAspect) / BAND_ASPECT;
}

/** Largest width whose locked-aspect box still fits in the frame. */
function maxWidth(srcAspect: number) {
  return Math.min(1, BAND_ASPECT / srcAspect);
}

function presetRect(corner: Corner, srcAspect: number): NormRect {
  const w = Math.min(0.28, maxWidth(srcAspect));
  const h = heightFor(w, srcAspect);
  const margin = 0.01;
  const x = corner === "tl" || corner === "bl" ? margin : 1 - w - margin;
  const y = corner === "tl" || corner === "tr" ? margin : 1 - h - margin;
  return { x: clamp(x, 0, 1 - w), y: clamp(y, 0, 1 - h), w, h };
}

/** Re-fits a stored rectangle to the locked aspect for this source, keeping its center. */
function normalize(rect: NormRect, srcAspect: number): NormRect {
  const w = clamp(rect.w, MIN_W, maxWidth(srcAspect));
  const h = heightFor(w, srcAspect);
  const cx = rect.x + rect.w / 2;
  const cy = rect.y + rect.h / 2;
  return { x: clamp(cx - w / 2, 0, 1 - w), y: clamp(cy - h / 2, 0, 1 - h), w, h };
}

function loadLastFacecam(): NormRect | null {
  try {
    const raw = localStorage.getItem(LAST_FACECAM_KEY);
    if (!raw) return null;
    const r = JSON.parse(raw);
    return [r.x, r.y, r.w, r.h].every((v) => typeof v === "number" && Number.isFinite(v)) ? r : null;
  } catch {
    return null;
  }
}

/** CSS background settings that show `rect` of the frame filling the element. */
function cropBackground(src: string, rect: NormRect): React.CSSProperties {
  const pos = (start: number, size: number) => (size >= 1 ? 0 : (start / (1 - size)) * 100);
  return {
    backgroundImage: `url(${src})`,
    backgroundSize: `${100 / rect.w}% ${100 / rect.h}%`,
    backgroundPosition: `${pos(rect.x, rect.w)}% ${pos(rect.y, rect.h)}%`,
  };
}

function gameplayRect(srcAspect: number): NormRect {
  // Centered full-height crop at the gameplay band's shape (mirrors the Rust side).
  if (srcAspect > GAMEPLAY_ASPECT) {
    const w = GAMEPLAY_ASPECT / srcAspect;
    return { x: (1 - w) / 2, y: 0, w, h: 1 };
  }
  const h = srcAspect / GAMEPLAY_ASPECT;
  return { x: 0, y: (1 - h) / 2, w: 1, h };
}

type Props = {
  projectId: string;
  sourceDuration: number | null;
  layout: ClipLayout;
  disabled: boolean;
  hasClips: boolean;
  onSaved: (layout: ClipLayout) => void;
};

export function LayoutPanel({ projectId, sourceDuration, layout, disabled, hasClips, onSaved }: Props) {
  const [frame, setFrame] = useState<string | null>(null);
  const [srcAspect, setSrcAspect] = useState(16 / 9);
  const [loadingFrame, setLoadingFrame] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [rect, setRect] = useState<NormRect | null>(layout.mode === "streamer" ? layout.facecam : null);
  const dragRef = useRef<DragState | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);

  const mode = layout.mode;

  // Pick up changes from the backend (e.g. after a refresh) unless the user is mid-drag.
  useEffect(() => {
    if (!dragRef.current) setRect(layout.mode === "streamer" ? layout.facecam : null);
  }, [layout]);

  useEffect(() => {
    if (mode === "streamer" && !frame && !loadingFrame) void loadFrame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, projectId]);

  async function loadFrame(atSec?: number) {
    setLoadingFrame(true);
    setError(null);
    try {
      const dataUrl = await invoke<string>("extract_preview_frame", { projectId, atSec: atSec ?? null });
      setFrame(dataUrl);
    } catch (err) {
      setError(`Couldn't load a preview frame: ${String(err)}`);
    } finally {
      setLoadingFrame(false);
    }
  }

  async function save(next: ClipLayout) {
    setSaveState("saving");
    setError(null);
    try {
      await invoke("set_project_layout", { projectId, layout: next });
      if (next.mode === "streamer") {
        try {
          localStorage.setItem(LAST_FACECAM_KEY, JSON.stringify(next.facecam));
        } catch {
          /* remembering the last position is only a convenience */
        }
      }
      onSaved(next);
      setSaveState("saved");
    } catch (err) {
      setError(String(err));
      setSaveState("idle");
    }
  }

  function chooseMode(next: "center" | "streamer") {
    if (next === mode) return;
    if (next === "center") {
      void save({ mode: "center" });
      return;
    }
    const start = normalize(loadLastFacecam() ?? presetRect("br", srcAspect), srcAspect);
    setRect(start);
    void save({ mode: "streamer", facecam: start });
  }

  function onImageLoad(event: React.SyntheticEvent<HTMLImageElement>) {
    const img = event.currentTarget;
    if (!img.naturalWidth || !img.naturalHeight) return;
    const aspect = img.naturalWidth / img.naturalHeight;
    setSrcAspect(aspect);
    // Stored rectangles may come from a source with a different shape; re-fit them once.
    if (rect) {
      const fitted = normalize(rect, aspect);
      const moved = ["x", "y", "w", "h"].some(
        (k) => Math.abs(fitted[k as keyof NormRect] - rect[k as keyof NormRect]) > 0.002,
      );
      setRect(fitted);
      if (moved) void save({ mode: "streamer", facecam: fitted });
    }
  }

  function pointerToNorm(event: React.PointerEvent) {
    const box = stageRef.current!.getBoundingClientRect();
    return {
      px: clamp((event.clientX - box.left) / box.width, 0, 1),
      py: clamp((event.clientY - box.top) / box.height, 0, 1),
    };
  }

  function startMove(event: React.PointerEvent) {
    if (!rect || disabled) return;
    event.preventDefault();
    stageRef.current?.setPointerCapture(event.pointerId);
    const { px, py } = pointerToNorm(event);
    dragRef.current = { kind: "move", startX: px, startY: py, origin: rect };
  }

  function startResize(event: React.PointerEvent, corner: Corner) {
    if (!rect || disabled) return;
    event.preventDefault();
    event.stopPropagation();
    stageRef.current?.setPointerCapture(event.pointerId);
    dragRef.current = { kind: "resize", corner, origin: rect };
  }

  function onPointerMove(event: React.PointerEvent) {
    const drag = dragRef.current;
    if (!drag) return;
    const { px, py } = pointerToNorm(event);
    const o = drag.origin;

    if (drag.kind === "move") {
      setRect({
        ...o,
        x: clamp(o.x + px - drag.startX, 0, 1 - o.w),
        y: clamp(o.y + py - drag.startY, 0, 1 - o.h),
      });
      return;
    }

    // Resize around the opposite (anchored) corner, keeping the band's shape.
    const left = drag.corner === "tl" || drag.corner === "bl";
    const top = drag.corner === "tl" || drag.corner === "tr";
    const anchorX = left ? o.x + o.w : o.x;
    const anchorY = top ? o.y + o.h : o.y;
    const roomX = left ? anchorX : 1 - anchorX;
    const roomY = top ? anchorY : 1 - anchorY;
    const widthFromY = (Math.abs(py - anchorY) * BAND_ASPECT) / srcAspect;
    const wanted = Math.max(Math.abs(px - anchorX), widthFromY);
    const limit = Math.min(roomX, (roomY * BAND_ASPECT) / srcAspect);
    const w = clamp(wanted, Math.min(MIN_W, limit), limit);
    const h = heightFor(w, srcAspect);
    setRect({ x: left ? anchorX - w : anchorX, y: top ? anchorY - h : anchorY, w, h });
  }

  function endDrag() {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (rect) void save({ mode: "streamer", facecam: rect });
  }

  function applyPreset(corner: Corner) {
    const next = presetRect(corner, srcAspect);
    setRect(next);
    void save({ mode: "streamer", facecam: next });
  }

  const game = gameplayRect(srcAspect);

  return (
    <section className="layout-panel">
      <div className="layout-panel-heading">
        <div>
          <h3>Clip Layout</h3>
          <p>
            {mode === "streamer"
              ? "Drag the box over the facecam. Drag its corners to resize."
              : "The whole frame is center-cropped to 9:16."}
          </p>
        </div>
        <div className="layout-toggle" role="radiogroup" aria-label="Clip layout">
          <button
            role="radio"
            aria-checked={mode === "center"}
            className={mode === "center" ? "active" : ""}
            onClick={() => chooseMode("center")}
            disabled={disabled}
          >
            <Crop size={14} /> Center crop
          </button>
          <button
            role="radio"
            aria-checked={mode === "streamer"}
            className={mode === "streamer" ? "active" : ""}
            onClick={() => chooseMode("streamer")}
            disabled={disabled}
          >
            <Gamepad2 size={14} /> Streamer
          </button>
        </div>
      </div>

      {mode === "streamer" && (
        <div className="layout-body">
          <div className="layout-source" style={{ maxWidth: `${Math.round(360 * srcAspect)}px` }}>
            <div
              ref={stageRef}
              className="layout-stage"
              style={{ aspectRatio: `${srcAspect}` }}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
            >
              {frame ? (
                <img src={frame} alt="Frame from the source video" onLoad={onImageLoad} draggable={false} />
              ) : (
                <div className="layout-stage-empty">
                  {loadingFrame ? <Loader2 className="spin" size={22} /> : <Video size={22} />}
                </div>
              )}
              {rect && (
                <div
                  className={`facecam-box ${disabled ? "locked" : ""}`}
                  style={{
                    left: `${rect.x * 100}%`,
                    top: `${rect.y * 100}%`,
                    width: `${rect.w * 100}%`,
                    height: `${rect.h * 100}%`,
                  }}
                  onPointerDown={startMove}
                >
                  <span className="facecam-label">Facecam</span>
                  {(["tl", "tr", "bl", "br"] as Corner[]).map((corner) => (
                    <span
                      key={corner}
                      className={`facecam-handle ${corner}`}
                      onPointerDown={(event) => startResize(event, corner)}
                    />
                  ))}
                </div>
              )}
            </div>

            <div className="layout-tools">
              <span>Quick place:</span>
              {(
                [
                  ["tl", "Top left"],
                  ["tr", "Top right"],
                  ["bl", "Bottom left"],
                  ["br", "Bottom right"],
                ] as [Corner, string][]
              ).map(([corner, label]) => (
                <button key={corner} onClick={() => applyPreset(corner)} disabled={disabled}>
                  {label}
                </button>
              ))}
              <button
                className="layout-new-frame"
                onClick={() => void loadFrame((0.1 + Math.random() * 0.8) * (sourceDuration ?? 60))}
                disabled={loadingFrame}
                title="Show a different moment from the video"
              >
                {loadingFrame ? <Loader2 className="spin" size={13} /> : <RefreshCw size={13} />} Another frame
              </button>
            </div>
          </div>

          <div className="layout-result">
            <div className="layout-phone" aria-label="Preview of the vertical clip">
              <div className="layout-phone-cam" style={frame && rect ? cropBackground(frame, rect) : undefined} />
              <div className="layout-phone-game" style={frame ? cropBackground(frame, game) : undefined} />
            </div>
            <div className="layout-save-state">
              {saveState === "saving" && (
                <>
                  <Loader2 className="spin" size={12} /> Saving…
                </>
              )}
              {saveState === "saved" && (
                <>
                  <Check size={12} /> Saved
                </>
              )}
            </div>
          </div>
        </div>
      )}

      <ClipExtras projectId={projectId} disabled={disabled} />
      <PostSettings projectId={projectId} disabled={disabled} />

      {hasClips && saveState === "saved" && (
        <p className="layout-hint">Clips you already cut keep their old layout. Click Re-cut to update them.</p>
      )}
      {error && <div className="render-log">{error}</div>}
    </section>
  );
}

export type WatermarkCorner = "top-left" | "top-right" | "bottom-left" | "bottom-right";

/** What the render step burns into each clip, besides captions. */
export type ClipBranding = {
  watermark: string | null;
  watermarkCorner: WatermarkCorner;
  adLabel: boolean;
};

const WATERMARK_KEY = "autoshorts_watermark";
const WATERMARK_CORNER_KEY = "autoshorts_watermark_corner";
const DEFAULT_WATERMARK = "@fullheatclips";
const adLabelKey = (projectId: string) => `autoshorts_ad_label_${projectId}`;

function readSetting(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeSetting(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* settings are a convenience; rendering still works without them */
  }
}

/** Current branding for a project: the watermark is shared, the AD label is per project. */
export function loadClipBranding(projectId: string): ClipBranding {
  const watermark = readSetting(WATERMARK_KEY) ?? DEFAULT_WATERMARK;
  const corner = (readSetting(WATERMARK_CORNER_KEY) as WatermarkCorner | null) ?? "top-right";
  return {
    watermark: watermark.trim() ? watermark : null,
    watermarkCorner: corner,
    adLabel: readSetting(adLabelKey(projectId)) === "true",
  };
}

/** Watermark and "AD" label settings, shown under the layout picker. */
export function ClipExtras({ projectId, disabled }: { projectId: string; disabled: boolean }) {
  const [watermark, setWatermark] = useState(() => readSetting(WATERMARK_KEY) ?? DEFAULT_WATERMARK);
  const [corner, setCorner] = useState<WatermarkCorner>(
    () => (readSetting(WATERMARK_CORNER_KEY) as WatermarkCorner | null) ?? "top-right",
  );
  const [adLabel, setAdLabel] = useState(() => readSetting(adLabelKey(projectId)) === "true");

  useEffect(() => setAdLabel(readSetting(adLabelKey(projectId)) === "true"), [projectId]);

  return (
    <div className="clip-extras">
      <label className="clip-extras-field">
        <span>Watermark</span>
        <input
          value={watermark}
          placeholder="@yourhandle (leave empty for none)"
          maxLength={40}
          disabled={disabled}
          onChange={(event) => {
            setWatermark(event.target.value);
            writeSetting(WATERMARK_KEY, event.target.value);
          }}
        />
      </label>
      <label className="clip-extras-field">
        <span>Corner</span>
        <select
          value={corner}
          disabled={disabled}
          onChange={(event) => {
            const next = event.target.value as WatermarkCorner;
            setCorner(next);
            writeSetting(WATERMARK_CORNER_KEY, next);
          }}
        >
          <option value="top-right">Top right</option>
          <option value="top-left">Top left</option>
          <option value="bottom-left">Bottom left</option>
          <option value="bottom-right">Bottom right</option>
        </select>
      </label>
      <label className="clip-extras-check" title="Shows 'AD' for the first 3 seconds. Many campaign briefs require it.">
        <input
          type="checkbox"
          checked={adLabel}
          disabled={disabled}
          onChange={(event) => {
            setAdLabel(event.target.checked);
            writeSetting(adLabelKey(projectId), String(event.target.checked));
          }}
        />
        <span>"AD" label for campaign clips (this project)</span>
      </label>
      <p className="clip-extras-hint">Applies to clips you cut or re-cut from now on.</p>
    </div>
  );
}

const streamerKey = (projectId: string) => `autoshorts_streamer_${projectId}`;
const campaignTagsKey = (projectId: string) => `autoshorts_campaign_tags_${projectId}`;

/** Who the clips are of and the hashtags the campaign requires; set per project. */
export function loadPostSettings(projectId: string): { streamerName: string | null; requiredHashtags: string | null } {
  const streamer = (readSetting(streamerKey(projectId)) ?? "").trim();
  const tags = (readSetting(campaignTagsKey(projectId)) ?? "").trim();
  return { streamerName: streamer || null, requiredHashtags: tags || null };
}

/** Streamer name and campaign hashtags used when AutoShorts writes post text. */
export function PostSettings({ projectId, disabled }: { projectId: string; disabled: boolean }) {
  const [streamer, setStreamer] = useState(() => readSetting(streamerKey(projectId)) ?? "");
  const [tags, setTags] = useState(() => readSetting(campaignTagsKey(projectId)) ?? "");

  useEffect(() => {
    setStreamer(readSetting(streamerKey(projectId)) ?? "");
    setTags(readSetting(campaignTagsKey(projectId)) ?? "");
  }, [projectId]);

  return (
    <div className="clip-extras">
      <label className="clip-extras-field">
        <span>Streamer name</span>
        <input
          value={streamer}
          placeholder="e.g. Lacy"
          maxLength={40}
          disabled={disabled}
          onChange={(event) => {
            setStreamer(event.target.value);
            writeSetting(streamerKey(projectId), event.target.value);
          }}
        />
      </label>
      <label className="clip-extras-field">
        <span>Campaign hashtags</span>
        <input
          value={tags}
          placeholder="e.g. #LACY"
          maxLength={80}
          disabled={disabled}
          onChange={(event) => {
            setTags(event.target.value);
            writeSetting(campaignTagsKey(projectId), event.target.value);
          }}
        />
      </label>
      <p className="clip-extras-hint">
        Used for the title, caption and hashtags AutoShorts writes for each clip (this project).
      </p>
    </div>
  );
}
