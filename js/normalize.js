// normalize.js — ComfyUI frontend extension entry point.
//
// Registers the "Export (API, normalized)" command, its menu entry, a
// keybinding and three settings. All transformation logic lives in
// normalize_core.js (no ComfyUI imports) so it can be unit-tested alone.
//
// The canvas graph is never modified: the extension only transforms the
// `output` object returned by app.graphToPrompt() and downloads it.
//
// Verified against ComfyUI_frontend 1.60 (2026-10). Relevant API history:
//   - commands / keybindings: v1.3.7, menuCommands: v1.3.1, settings: v1.3.22
//   - top-level menu group "Workflow" was renamed to "File" in v1.27.0
//   - kind-based toast API (toast.success/error) since v1.58; toast.add() still works

import { app } from "../../scripts/app.js";
import { MODES, normalizePrompt } from "./normalize_core.js";

const EXTENSION_NAME = "Comfy.NormalizedExport";
const COMMAND_ID = "NormalizedExport.ExportApiNormalized";

const SETTING_MODE = "NormalizedExport.OrderMode";
const SETTING_STRIP = "NormalizedExport.StripTitlePrefix";
const SETTING_FILENAME = "NormalizedExport.Filename";

const DEFAULT_MODE = "title";
const DEFAULT_FILENAME = "workflow_api_normalized.json";

const SETTINGS_CATEGORY = "Normalized Export";

/** Read a setting through the current API, falling back to the legacy one. */
function getSetting(id, fallback) {
  let value;
  try {
    value = app.extensionManager?.setting?.get(id);
  } catch (_) {
    value = undefined;
  }
  if (value === undefined && typeof app.ui?.settings?.getSettingValue === "function") {
    value = app.ui.settings.getSettingValue(id);
  }
  return value === undefined || value === null ? fallback : value;
}

/** Show a toast using the kind-based API when present, else the legacy add(). */
function notify(severity, title, description) {
  const toast = app.extensionManager?.toast;
  if (!toast) {
    (severity === "error" ? console.error : console.log)(`[NormalizedExport] ${title}`, description ?? "");
    return;
  }
  if (typeof toast[severity] === "function") {
    toast[severity](title, { description, duration: severity === "error" ? 6000 : 3000 });
    return;
  }
  toast.add({ severity, summary: title, detail: description, life: severity === "error" ? 6000 : 3000 });
}

/** Trigger a browser download of a JSON string. */
function downloadJson(filename, text) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke after the click has been dispatched.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function resolveFilename() {
  let name = String(getSetting(SETTING_FILENAME, DEFAULT_FILENAME)).trim();
  if (!name) name = DEFAULT_FILENAME;
  if (!/\.json$/i.test(name)) name += ".json";
  return name;
}

function resolveMode() {
  const mode = getSetting(SETTING_MODE, DEFAULT_MODE);
  return MODES.includes(mode) ? mode : DEFAULT_MODE;
}

async function exportNormalized() {
  let result;
  try {
    const graphData = await app.graphToPrompt();
    const output = graphData?.output;
    if (!output || typeof output !== "object") {
      throw new Error("graphToPrompt() returned no API output");
    }

    const mode = resolveMode();
    const stripPrefix = Boolean(getSetting(SETTING_STRIP, false));
    result = normalizePrompt(output, { mode, stripPrefix });

    const filename = resolveFilename();
    downloadJson(filename, JSON.stringify(result.prompt, null, 2));

    const count = result.order.length;
    notify("success", "Exported normalized API workflow", `${filename} · ${count} node${count === 1 ? "" : "s"} · mode: ${mode}`);
  } catch (err) {
    console.error("[NormalizedExport] export failed", err);
    notify("error", "Normalized export failed", err?.message ?? String(err));
  }
}

/**
 * Pick the top-level menu group that holds the core "Export (API)" entry.
 * Frontend >= 1.27 uses "File"; older versions used "Workflow".
 */
function exportMenuPath() {
  const version = globalThis.__COMFYUI_FRONTEND_VERSION__;
  const major = /^(\d+)\.(\d+)/.exec(String(version ?? ""));
  if (major) {
    const [, maj, min] = major.map(Number);
    if (maj < 1 || (maj === 1 && min < 27)) return ["Workflow"];
  }
  return ["File"];
}

app.registerExtension({
  name: EXTENSION_NAME,

  settings: [
    {
      id: SETTING_MODE,
      name: "Node order mode",
      type: "combo",
      defaultValue: DEFAULT_MODE,
      options: [
        { text: "title — [n] prefix first, then topological", value: "title" },
        { text: "sequential — original id order", value: "sequential" },
        { text: "topological — execution order", value: "topological" },
      ],
      category: [SETTINGS_CATEGORY, "Ordering", "Node order mode"],
      tooltip:
        "title: nodes whose title starts with [n] come first (ascending n), the rest in topological order. " +
        "sequential: original node id order. topological: Kahn order, ties broken by original id.",
    },
    {
      id: SETTING_STRIP,
      name: "Strip [n] prefix from titles",
      type: "boolean",
      defaultValue: false,
      category: [SETTINGS_CATEGORY, "Ordering", "Strip [n] prefix from titles"],
      tooltip: "Remove the leading [n] marker from _meta.title in the exported file.",
    },
    {
      id: SETTING_FILENAME,
      name: "Download filename",
      type: "text",
      defaultValue: DEFAULT_FILENAME,
      category: [SETTINGS_CATEGORY, "Output", "Download filename"],
      tooltip: "Name of the downloaded file. '.json' is appended if missing.",
    },
  ],

  commands: [
    {
      id: COMMAND_ID,
      icon: "icon-[lucide--download]",
      label: "Export Workflow (API Format, normalized)",
      menubarLabel: "Export (API, normalized)",
      tooltip: "Export the API-format workflow with node ids renumbered 1..N",
      versionAdded: "1.0.0",
      function: exportNormalized,
    },
  ],

  menuCommands: [{ path: exportMenuPath(), commands: [COMMAND_ID] }],

  // The menu location moved between frontend versions, so also expose a
  // shortcut. Ctrl matches Cmd on macOS in the frontend's key matching.
  keybindings: [
    {
      combo: { key: "e", ctrl: true, alt: true },
      commandId: COMMAND_ID,
    },
  ],
});
