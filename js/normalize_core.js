/**
 * normalize_core.js
 *
 * Pure, browser-compatible ES module that renumbers the node ids of a
 * ComfyUI "API format" prompt into a dense "1".."N" sequence.
 *
 * A prompt is an object keyed by node id strings. Each value looks like
 *   { class_type: string, inputs: { [name]: value }, _meta?: { title?: string } }
 * An input value is a *link* iff it is an array of length 2 whose first
 * element (compared as String) is a key of the prompt; the second element is
 * the output slot index of the source node.
 *
 * Nodes that live inside subgraphs are flattened to path keys such as
 * "12:37" or "12:37:5", so ids are compared segment-wise (see compareIds).
 *
 * No dependencies, no Node builtins, no ComfyUI imports.
 */

/** Supported ordering modes, in the order they are documented. */
export const MODES = ['title', 'sequential', 'topological'];

/**
 * Compare two node ids segment-wise.
 * Ids are split on ':'; each segment is compared numerically when both parse
 * as numbers, otherwise as strings. A shorter id that is a prefix of a longer
 * one sorts first. Example order: 5 < 12 < 12:3 < 12:37 < 13.
 * @param {string|number} a
 * @param {string|number} b
 * @returns {-1|0|1}
 */
export function compareIds(a, b) {
  const sa = String(a).split(':');
  const sb = String(b).split(':');
  const len = Math.min(sa.length, sb.length);
  for (let i = 0; i < len; i++) {
    const na = Number(sa[i]);
    const nb = Number(sb[i]);
    const bothNumeric = sa[i] !== '' && sb[i] !== '' && !Number.isNaN(na) && !Number.isNaN(nb);
    if (bothNumeric) {
      if (na < nb) return -1;
      if (na > nb) return 1;
    } else if (sa[i] !== sb[i]) {
      return sa[i] < sb[i] ? -1 : 1;
    }
  }
  if (sa.length === sb.length) return 0;
  return sa.length < sb.length ? -1 : 1;
}

/** Leading "[n]" title prefix, e.g. "[3] KSampler" or "[ 7 ]Foo". */
const TITLE_PREFIX_RE = /^\s*\[\s*(\d+)\s*\]/;

/**
 * Parse the explicit ordering number from a node title.
 * @param {unknown} title
 * @returns {number|null} the integer n for a leading "[n]", otherwise null
 */
export function parseTitleOrder(title) {
  if (typeof title !== 'string') return null;
  const m = TITLE_PREFIX_RE.exec(title);
  return m ? parseInt(m[1], 10) : null;
}

/**
 * Remove a leading "[n]" prefix (and the whitespace that follows it).
 * "[3] KSampler" -> "KSampler", "[3]" -> "", "KSampler" -> "KSampler".
 * @param {string} title
 * @returns {string}
 */
export function stripTitlePrefix(title) {
  if (typeof title !== 'string') return title;
  return title.replace(TITLE_PREFIX_RE, '').replace(/^\s+/, '');
}

/**
 * Whether an input value is a link to another node of the given prompt.
 * @param {unknown} value
 * @param {object} prompt
 * @returns {boolean}
 */
export function isLink(value, prompt) {
  if (!Array.isArray(value) || value.length !== 2) return false;
  const src = value[0];
  if (typeof src !== 'string' && typeof src !== 'number') return false;
  return Object.prototype.hasOwnProperty.call(prompt, String(src));
}

/**
 * Iterate over the inputs of a node, yielding [name, value] pairs.
 * Tolerates nodes without an `inputs` object.
 * @param {object} node
 * @returns {Array<[string, unknown]>}
 */
function inputEntries(node) {
  const inputs = node && node.inputs;
  if (!inputs || typeof inputs !== 'object') return [];
  return Object.entries(inputs);
}

/**
 * Topological order of the prompt (Kahn's algorithm).
 * Each link input [src, slot] of a node is an edge src -> node. Among the
 * ready nodes the smallest one (by compareIds) is always picked, so the
 * result is deterministic and independent of key insertion order. Nodes
 * left over because of cycles are appended, sorted by compareIds.
 * @param {object} prompt
 * @returns {string[]} original ids
 */
export function topologicalOrder(prompt) {
  const ids = Object.keys(prompt);
  const indegree = new Map(ids.map((id) => [id, 0]));
  const dependents = new Map(ids.map((id) => [id, []]));

  for (const id of ids) {
    for (const [, value] of inputEntries(prompt[id])) {
      if (!isLink(value, prompt)) continue;
      const src = String(value[0]);
      indegree.set(id, indegree.get(id) + 1);
      dependents.get(src).push(id);
    }
  }

  // Ready set, kept sorted by compareIds so the smallest id pops first.
  const ready = ids.filter((id) => indegree.get(id) === 0).sort(compareIds);
  const result = [];
  const placed = new Set();

  while (ready.length > 0) {
    const id = ready.shift();
    result.push(id);
    placed.add(id);
    for (const dep of dependents.get(id)) {
      const d = indegree.get(dep) - 1;
      indegree.set(dep, d);
      if (d === 0) ready.push(dep);
    }
    ready.sort(compareIds);
  }

  // Anything not placed is part of (or downstream of) a cycle.
  if (result.length < ids.length) {
    const leftover = ids.filter((id) => !placed.has(id)).sort(compareIds);
    result.push(...leftover);
  }
  return result;
}

/**
 * All node ids sorted by compareIds.
 * @param {object} prompt
 * @returns {string[]} original ids
 */
export function sequentialOrder(prompt) {
  return Object.keys(prompt).sort(compareIds);
}

/**
 * Title-driven order: nodes whose `_meta.title` starts with "[n]" come first,
 * sorted by n ascending (ties broken by topological position), followed by
 * every other node in topological order.
 * @param {object} prompt
 * @returns {string[]} original ids
 */
export function titleOrder(prompt) {
  const topo = topologicalOrder(prompt);
  const topoIndex = new Map(topo.map((id, i) => [id, i]));

  const numbered = [];
  const rest = [];
  for (const id of topo) {
    const node = prompt[id];
    const n = parseTitleOrder(node && node._meta && node._meta.title);
    if (n === null) rest.push(id);
    else numbered.push({ id, n });
  }
  numbered.sort((a, b) => a.n - b.n || topoIndex.get(a.id) - topoIndex.get(b.id));

  return [...numbered.map((e) => e.id), ...rest];
}

/**
 * Compute the node order for the given mode.
 * @param {object} prompt
 * @param {'title'|'sequential'|'topological'} [mode='title']
 * @returns {string[]} original ids
 * @throws {Error} on an unknown mode
 */
export function computeOrder(prompt, mode = 'title') {
  switch (mode) {
    case 'title':
      return titleOrder(prompt);
    case 'sequential':
      return sequentialOrder(prompt);
    case 'topological':
      return topologicalOrder(prompt);
    default:
      throw new Error(`Unknown order mode: ${String(mode)} (expected one of ${MODES.join(', ')})`);
  }
}

/**
 * Map each original id (in order) to its new dense id "1".."N".
 * @param {string[]} order
 * @returns {{ [oldId: string]: string }}
 */
export function buildIdMap(order) {
  const idMap = {};
  order.forEach((id, i) => {
    idMap[String(id)] = String(i + 1);
  });
  return idMap;
}

/**
 * Produce a copy of the prompt whose node ids are renumbered "1".."N".
 * The input is never mutated. Links are rewritten to the new ids; any other
 * value (including non-link arrays) is kept as is.
 * @param {object} prompt
 * @param {{ mode?: 'title'|'sequential'|'topological', stripPrefix?: boolean }} [options]
 * @returns {{ prompt: object, idMap: { [oldId: string]: string }, order: string[] }}
 */
export function normalizePrompt(prompt, options = {}) {
  const { mode = 'title', stripPrefix = false } = options;
  const clone = structuredClone(prompt);
  const order = computeOrder(clone, mode);
  const idMap = buildIdMap(order);

  const newPrompt = {};
  for (const oldId of order) {
    const node = clone[oldId];
    for (const [name, value] of inputEntries(node)) {
      // Link detection is done against the original key set.
      if (isLink(value, prompt)) {
        node.inputs[name] = [idMap[String(value[0])], value[1]];
      }
    }
    if (stripPrefix && node && node._meta && typeof node._meta.title === 'string') {
      node._meta.title = stripTitlePrefix(node._meta.title);
    }
    newPrompt[idMap[oldId]] = node;
  }

  return { prompt: newPrompt, idMap, order };
}
