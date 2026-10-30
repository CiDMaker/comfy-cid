import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import {
  MODES,
  compareIds,
  parseTitleOrder,
  stripTitlePrefix,
  isLink,
  topologicalOrder,
  sequentialOrder,
  titleOrder,
  computeOrder,
  buildIdMap,
  normalizePrompt,
} from '../js/normalize_core.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Standard ComfyUI txt2img API prompt. Re-created on every call. */
function txt2img() {
  return {
    '3': {
      class_type: 'KSampler',
      inputs: {
        seed: 42,
        steps: 20,
        cfg: 7,
        sampler_name: 'euler',
        scheduler: 'normal',
        denoise: 1,
        model: ['4', 0],
        positive: ['6', 0],
        negative: ['7', 0],
        latent_image: ['5', 0],
      },
      _meta: { title: 'KSampler' },
    },
    '4': {
      class_type: 'CheckpointLoaderSimple',
      inputs: { ckpt_name: 'model.safetensors' },
      _meta: { title: 'Load Checkpoint' },
    },
    '5': {
      class_type: 'EmptyLatentImage',
      inputs: { width: 512, height: 512, batch_size: 1 },
      _meta: { title: 'Empty Latent Image' },
    },
    '6': {
      class_type: 'CLIPTextEncode',
      inputs: { text: 'a cat', clip: ['4', 1] },
      _meta: { title: 'CLIP Text Encode (Prompt)' },
    },
    '7': {
      class_type: 'CLIPTextEncode',
      inputs: { text: 'blurry', clip: ['4', 1] },
      _meta: { title: 'CLIP Text Encode (Negative)' },
    },
    '8': {
      class_type: 'VAEDecode',
      inputs: { samples: ['3', 0], vae: ['4', 2] },
      _meta: { title: 'VAE Decode' },
    },
    '9': {
      class_type: 'SaveImage',
      inputs: { filename_prefix: 'ComfyUI', images: ['8', 0] },
      _meta: { title: 'Save Image' },
    },
  };
}

/** Subgraph-style fixture with path keys. */
function subgraph() {
  return {
    '1': { class_type: 'A', inputs: { v: 1 }, _meta: { title: 'A' } },
    '12:3': { class_type: 'B', inputs: { v: 2 }, _meta: { title: 'B' } },
    '12:37': { class_type: 'C', inputs: { src: ['12:3', 0] }, _meta: { title: 'C' } },
    '13': { class_type: 'D', inputs: { src: ['12:37', 0] }, _meta: { title: 'D' } },
    '2': { class_type: 'E', inputs: { src: ['1', 0] }, _meta: { title: 'E' } },
  };
}

/** Two-node cycle plus one independent node. */
function cyclic() {
  return {
    '10': { class_type: 'X', inputs: { a: ['20', 0] }, _meta: { title: 'X' } },
    '20': { class_type: 'Y', inputs: { a: ['10', 0] }, _meta: { title: 'Y' } },
    '5': { class_type: 'Z', inputs: { v: 1 }, _meta: { title: 'Z' } },
  };
}

const FIXTURES = { txt2img, subgraph, cyclic };

/** Count links in a prompt (an input value that isLink against that prompt). */
function countLinks(prompt) {
  let n = 0;
  for (const node of Object.values(prompt)) {
    for (const value of Object.values(node.inputs || {})) {
      if (isLink(value, prompt)) n += 1;
    }
  }
  return n;
}

/** Shared link-integrity assertions for a normalizePrompt result. */
function assertIntegrity(input, result) {
  const { prompt: out, idMap, order } = result;
  const outKeys = Object.keys(out);
  const inKeys = Object.keys(input);

  // Same node count, dense ids "1".."N" in insertion order.
  assert.equal(outKeys.length, inKeys.length);
  assert.deepEqual(outKeys, inKeys.map((_, i) => String(i + 1)));
  assert.deepEqual(order.slice().sort(), inKeys.slice().sort());

  // Every link in the output points to an existing output key.
  for (const node of Object.values(out)) {
    for (const value of Object.values(node.inputs || {})) {
      if (isLink(value, out)) {
        assert.ok(Object.prototype.hasOwnProperty.call(out, String(value[0])));
      }
    }
  }
  assert.equal(countLinks(out), countLinks(input));

  // Mapping consistency: node at new id has the same class_type as original.
  const inverse = Object.fromEntries(Object.entries(idMap).map(([k, v]) => [v, k]));
  for (const newId of outKeys) {
    const oldId = inverse[newId];
    assert.ok(oldId !== undefined, `no inverse mapping for ${newId}`);
    assert.equal(out[newId].class_type, input[oldId].class_type);
  }
  // Every rewritten link maps back to the original source.
  for (const newId of outKeys) {
    const oldId = inverse[newId];
    for (const [name, value] of Object.entries(input[oldId].inputs || {})) {
      if (isLink(value, input)) {
        assert.deepEqual(out[newId].inputs[name], [idMap[String(value[0])], value[1]]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 1. compareIds
// ---------------------------------------------------------------------------

describe('compareIds', () => {
  test('sorts segment-wise numerically', () => {
    const sorted = ['13', '12:37', '5', '12', '12:3'].sort(compareIds);
    assert.deepEqual(sorted, ['5', '12', '12:3', '12:37', '13']);
  });

  test('equal ids return 0', () => {
    assert.equal(compareIds('12:3', '12:3'), 0);
    assert.equal(compareIds('5', '5'), 0);
  });

  test('numeric and string keys compare equally', () => {
    assert.equal(compareIds('5', 5), 0);
    assert.equal(compareIds(5, '12'), -1);
    assert.equal(compareIds('12', 5), 1);
  });

  test('nested paths and non-numeric segments', () => {
    assert.equal(compareIds('12:37', '12:37:5'), -1);
    assert.equal(compareIds('12:37:5', '12:37'), 1);
    assert.equal(compareIds('a', 'b'), -1);
    assert.equal(compareIds('b', 'a'), 1);
  });
});

// ---------------------------------------------------------------------------
// Title helpers
// ---------------------------------------------------------------------------

describe('parseTitleOrder / stripTitlePrefix', () => {
  test('parses leading [n]', () => {
    assert.equal(parseTitleOrder('[3] KSampler'), 3);
    assert.equal(parseTitleOrder('[ 7 ]Foo'), 7);
    assert.equal(parseTitleOrder('  [10]'), 10);
  });

  test('does not parse non-leading or missing prefix', () => {
    assert.equal(parseTitleOrder('Foo [3]'), null);
    assert.equal(parseTitleOrder('KSampler'), null);
    assert.equal(parseTitleOrder(''), null);
    assert.equal(parseTitleOrder(undefined), null);
    assert.equal(parseTitleOrder(42), null);
  });

  test('strips prefix', () => {
    assert.equal(stripTitlePrefix('[3] KSampler'), 'KSampler');
    assert.equal(stripTitlePrefix('[3]'), '');
    assert.equal(stripTitlePrefix('[ 7 ]Foo'), 'Foo');
    assert.equal(stripTitlePrefix('KSampler'), 'KSampler');
    assert.equal(stripTitlePrefix('Foo [3]'), 'Foo [3]');
  });
});

// ---------------------------------------------------------------------------
// isLink
// ---------------------------------------------------------------------------

describe('isLink', () => {
  test('detects links by key presence', () => {
    const p = txt2img();
    assert.equal(isLink(['4', 0], p), true);
    assert.equal(isLink([4, 0], p), true);
    assert.equal(isLink(['999', 0], p), false);
    assert.equal(isLink([1, 2, 3], p), false);
    assert.equal(isLink(['4'], p), false);
    assert.equal(isLink('4', p), false);
    assert.equal(isLink(null, p), false);
  });
});

// ---------------------------------------------------------------------------
// 2. sequential
// ---------------------------------------------------------------------------

describe('sequential mode', () => {
  test('txt2img maps 3..9 to 1..7', () => {
    const input = txt2img();
    const { prompt, idMap, order } = normalizePrompt(input, { mode: 'sequential' });
    assert.deepEqual(order, ['3', '4', '5', '6', '7', '8', '9']);
    assert.deepEqual(idMap, { 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7' });
    assert.deepEqual(Object.keys(prompt), ['1', '2', '3', '4', '5', '6', '7']);
    assert.equal(prompt['1'].class_type, 'KSampler');
    assert.deepEqual(prompt['1'].inputs.model, ['2', 0]);
    assert.deepEqual(prompt['7'].inputs.images, ['6', 0]);
    assertIntegrity(input, { prompt, idMap, order });
  });
});

// ---------------------------------------------------------------------------
// 3. topological
// ---------------------------------------------------------------------------

describe('topological mode', () => {
  test('txt2img order is 4,5,6,7,3,8,9', () => {
    const input = txt2img();
    assert.deepEqual(topologicalOrder(input), ['4', '5', '6', '7', '3', '8', '9']);

    const result = normalizePrompt(input, { mode: 'topological' });
    assert.deepEqual(result.order, ['4', '5', '6', '7', '3', '8', '9']);
    assert.equal(result.idMap['3'], '5');
    assert.equal(result.idMap['8'], '6');
    assert.equal(result.idMap['9'], '7');
    assert.equal(result.prompt['1'].class_type, 'CheckpointLoaderSimple');
    assert.equal(result.prompt['5'].class_type, 'KSampler');
    assert.deepEqual(result.prompt['5'].inputs.model, ['1', 0]);
    assert.deepEqual(result.prompt['5'].inputs.positive, ['3', 0]);
    assert.deepEqual(result.prompt['5'].inputs.negative, ['4', 0]);
    assert.deepEqual(result.prompt['5'].inputs.latent_image, ['2', 0]);
    assert.deepEqual(result.prompt['6'].inputs.samples, ['5', 0]);
    assert.deepEqual(result.prompt['6'].inputs.vae, ['1', 2]);
    assert.deepEqual(result.prompt['7'].inputs.images, ['6', 0]);
    assertIntegrity(input, result);
  });
});

// ---------------------------------------------------------------------------
// 4 & 5. title mode
// ---------------------------------------------------------------------------

describe('title mode', () => {
  test('without [n] titles equals topological', () => {
    const input = txt2img();
    const a = normalizePrompt(input, { mode: 'title' });
    const b = normalizePrompt(input, { mode: 'topological' });
    assert.deepEqual(a, b);
    assert.deepEqual(titleOrder(input), topologicalOrder(input));
  });

  test('is the default mode', () => {
    const input = txt2img();
    assert.deepEqual(normalizePrompt(input), normalizePrompt(input, { mode: 'title' }));
    assert.deepEqual(computeOrder(input), titleOrder(input));
  });

  test('[n] titles come first, ties broken topologically', () => {
    const input = txt2img();
    input['9']._meta.title = '[1] Save';
    input['5']._meta.title = '[10] Latent';
    input['7']._meta.title = '[10] Neg';
    input['6']._meta.title = '[20] Pos';

    const order = titleOrder(input);
    assert.deepEqual(order, ['9', '5', '7', '6', '4', '3', '8']);

    // Tie between [10] Latent (5) and [10] Neg (7) is broken by topological
    // position: 5 precedes 7 in topologicalOrder.
    const topo = topologicalOrder(input);
    assert.ok(topo.indexOf('5') < topo.indexOf('7'));
    assert.ok(order.indexOf('5') < order.indexOf('7'));

    const result = normalizePrompt(input, { mode: 'title' });
    assert.deepEqual(result.order, order);
    assert.equal(result.prompt['1'].class_type, 'SaveImage');
    assert.equal(result.prompt['2'].class_type, 'EmptyLatentImage');
    assert.equal(result.prompt['3']._meta.title, '[10] Neg');
    assert.equal(result.prompt['4']._meta.title, '[20] Pos');
    assertIntegrity(input, result);
  });

  test('tie order flips when topological order flips', () => {
    // Give the two [10] nodes a dependency so topological position decides.
    const input = txt2img();
    input['5']._meta.title = '[10] Latent';
    input['7']._meta.title = '[10] Neg';
    // Make 5 depend on 7 (artificial, but valid for ordering purposes).
    input['5'].inputs.extra = ['7', 0];
    const order = titleOrder(input);
    assert.ok(order.indexOf('7') < order.indexOf('5'));
    assert.deepEqual(order.slice(0, 2), ['7', '5']);
  });

  test('"[ 7 ]Foo" parses as 7; "Foo [3]" is not a prefix', () => {
    const input = txt2img();
    input['8']._meta.title = '[ 7 ]Foo';
    input['4']._meta.title = 'Foo [3]';
    assert.equal(parseTitleOrder(input['8']._meta.title), 7);
    assert.equal(parseTitleOrder(input['4']._meta.title), null);
    const order = titleOrder(input);
    assert.equal(order[0], '8');
    // Remaining nodes in topological order (minus 8).
    assert.deepEqual(order.slice(1), topologicalOrder(input).filter((id) => id !== '8'));
  });

  test('gaps in n are fine', () => {
    const input = txt2img();
    input['3']._meta.title = '[30] S';
    input['4']._meta.title = '[10] L';
    input['9']._meta.title = '[20] Save';
    assert.deepEqual(titleOrder(input).slice(0, 3), ['4', '9', '3']);
  });
});

// ---------------------------------------------------------------------------
// 6. Subgraph path keys
// ---------------------------------------------------------------------------

describe('subgraph path keys', () => {
  test('sequential order', () => {
    assert.deepEqual(sequentialOrder(subgraph()), ['1', '2', '12:3', '12:37', '13']);
  });

  test('topological order is sensible and deterministic', () => {
    const input = subgraph();
    const topo = topologicalOrder(input);
    assert.deepEqual(topo, ['1', '2', '12:3', '12:37', '13']);
    assert.ok(topo.indexOf('12:3') < topo.indexOf('12:37'));
    assert.ok(topo.indexOf('12:37') < topo.indexOf('13'));
    assert.ok(topo.indexOf('1') < topo.indexOf('2'));
    assert.deepEqual(topologicalOrder(subgraph()), topo);
  });

  for (const mode of MODES) {
    test(`output keys are dense without ':' and links updated (${mode})`, () => {
      const input = subgraph();
      const result = normalizePrompt(input, { mode });
      const keys = Object.keys(result.prompt);
      assert.deepEqual(keys, ['1', '2', '3', '4', '5']);
      for (const k of keys) assert.ok(!k.includes(':'));
      for (const node of Object.values(result.prompt)) {
        for (const value of Object.values(node.inputs)) {
          if (Array.isArray(value)) assert.ok(!String(value[0]).includes(':'));
        }
      }
      // Specific link checks.
      const { idMap, prompt } = result;
      assert.deepEqual(prompt[idMap['12:37']].inputs.src, [idMap['12:3'], 0]);
      assert.deepEqual(prompt[idMap['13']].inputs.src, [idMap['12:37'], 0]);
      assert.deepEqual(prompt[idMap['2']].inputs.src, [idMap['1'], 0]);
      assertIntegrity(input, result);
    });
  }
});

// ---------------------------------------------------------------------------
// 7. Link integrity across all modes and fixtures
// ---------------------------------------------------------------------------

describe('link integrity', () => {
  for (const [name, make] of Object.entries(FIXTURES)) {
    for (const mode of MODES) {
      test(`${name} / ${mode}`, () => {
        const input = make();
        assertIntegrity(input, normalizePrompt(input, { mode }));
      });
    }
  }
});

// ---------------------------------------------------------------------------
// 8. Non-link arrays untouched
// ---------------------------------------------------------------------------

describe('non-link arrays', () => {
  for (const mode of MODES) {
    test(`left untouched (${mode})`, () => {
      const input = txt2img();
      input['5'].inputs.not_a_link = ['999', 0];
      input['5'].inputs.triple = [1, 2, 3];
      input['5'].inputs.single = ['4'];
      input['5'].inputs.nested = [['4', 0]];
      const { prompt, idMap } = normalizePrompt(input, { mode });
      const node = prompt[idMap['5']];
      assert.deepEqual(node.inputs.not_a_link, ['999', 0]);
      assert.deepEqual(node.inputs.triple, [1, 2, 3]);
      assert.deepEqual(node.inputs.single, ['4']);
      assert.deepEqual(node.inputs.nested, [['4', 0]]);
      assertIntegrity(input, normalizePrompt(input, { mode }));
    });
  }

  test('numeric source ids are rewritten as strings', () => {
    const input = txt2img();
    input['6'].inputs.clip = [4, 1];
    const { prompt, idMap } = normalizePrompt(input, { mode: 'sequential' });
    assert.deepEqual(prompt[idMap['6']].inputs.clip, [idMap['4'], 1]);
    assert.equal(typeof prompt[idMap['6']].inputs.clip[0], 'string');
  });
});

// ---------------------------------------------------------------------------
// 9. Input immutability
// ---------------------------------------------------------------------------

describe('input immutability', () => {
  for (const mode of MODES) {
    test(`input is not mutated (${mode})`, () => {
      const input = txt2img();
      input['9']._meta.title = '[1] Save';
      const before = JSON.stringify(input);
      const result = normalizePrompt(input, { mode, stripPrefix: true });
      assert.equal(JSON.stringify(input), before);
      assert.notEqual(result.prompt, input);
      for (const node of Object.values(result.prompt)) {
        for (const orig of Object.values(input)) {
          assert.notEqual(node, orig);
          assert.notEqual(node.inputs, orig.inputs);
        }
      }
    });
  }
});

// ---------------------------------------------------------------------------
// 10. Determinism
// ---------------------------------------------------------------------------

describe('determinism', () => {
  /** Rebuild an object with its keys inserted in a different order. */
  function shuffleKeys(obj, keys) {
    const out = {};
    for (const k of keys) out[k] = obj[k];
    assert.equal(Object.keys(out).length, Object.keys(obj).length);
    return out;
  }

  for (const [name, make] of Object.entries(FIXTURES)) {
    for (const mode of MODES) {
      test(`${name} / ${mode}: same input twice, and shuffled insertion`, () => {
        const input = make();
        const a = normalizePrompt(input, { mode });
        const b = normalizePrompt(input, { mode });
        assert.deepEqual(a, b);

        // Note: integer-like keys are always enumerated in ascending numeric
        // order by JS, so shuffling only affects path keys like "12:3"; the
        // result must be identical regardless.
        const keys = Object.keys(input).reverse();
        const shuffled = shuffleKeys(input, keys);
        const c = normalizePrompt(shuffled, { mode });
        assert.deepEqual(JSON.parse(JSON.stringify(c)), JSON.parse(JSON.stringify(a)));
        assert.deepEqual(Object.keys(c.prompt), Object.keys(a.prompt));
      });
    }
  }

  test('path keys inserted in shuffled order yield the same output', () => {
    const base = subgraph();
    const reordered = {};
    for (const k of ['13', '12:37', '2', '12:3', '1']) reordered[k] = base[k];
    for (const mode of MODES) {
      const a = normalizePrompt(base, { mode });
      const b = normalizePrompt(reordered, { mode });
      assert.deepEqual(a.order, b.order);
      assert.deepEqual(a.idMap, b.idMap);
      assert.deepEqual(JSON.stringify(a.prompt), JSON.stringify(b.prompt));
    }
  });
});

// ---------------------------------------------------------------------------
// 11. stripPrefix
// ---------------------------------------------------------------------------

describe('stripPrefix option', () => {
  function titled() {
    const input = txt2img();
    input['9']._meta.title = '[1] Save';
    delete input['5']._meta; // node without _meta
    return input;
  }

  test('false leaves prefixed title intact', () => {
    const input = titled();
    const { prompt, idMap } = normalizePrompt(input, { stripPrefix: false });
    assert.equal(prompt[idMap['9']]._meta.title, '[1] Save');
  });

  test('default is false', () => {
    const input = titled();
    const { prompt, idMap } = normalizePrompt(input);
    assert.equal(prompt[idMap['9']]._meta.title, '[1] Save');
  });

  test('true strips the prefix and leaves others unchanged', () => {
    const input = titled();
    const { prompt, idMap } = normalizePrompt(input, { stripPrefix: true });
    assert.equal(prompt[idMap['9']]._meta.title, 'Save');
    assert.equal(prompt[idMap['3']]._meta.title, 'KSampler');
    assert.equal(prompt[idMap['4']]._meta.title, 'Load Checkpoint');
    // Node without _meta: no _meta created.
    assert.equal(Object.prototype.hasOwnProperty.call(prompt[idMap['5']], '_meta'), false);
    // Input still intact.
    assert.equal(input['9']._meta.title, '[1] Save');
  });

  test('_meta without a string title is untouched', () => {
    const input = txt2img();
    input['4']._meta = { other: 1 };
    input['6']._meta = { title: 7 };
    const { prompt, idMap } = normalizePrompt(input, { stripPrefix: true });
    assert.deepEqual(prompt[idMap['4']]._meta, { other: 1 });
    assert.deepEqual(prompt[idMap['6']]._meta, { title: 7 });
  });
});

// ---------------------------------------------------------------------------
// 12. Errors and empty prompt
// ---------------------------------------------------------------------------

describe('computeOrder / edge cases', () => {
  test('throws on unknown mode', () => {
    assert.throws(() => computeOrder(txt2img(), 'bogus'), /Unknown order mode/);
    assert.throws(() => normalizePrompt(txt2img(), { mode: 'bogus' }), /Unknown order mode/);
  });

  test('MODES lists the supported modes', () => {
    assert.deepEqual(MODES, ['title', 'sequential', 'topological']);
    for (const mode of MODES) assert.doesNotThrow(() => computeOrder(txt2img(), mode));
  });

  test('empty prompt', () => {
    for (const mode of MODES) {
      const result = normalizePrompt({}, { mode });
      assert.deepEqual(result, { prompt: {}, idMap: {}, order: [] });
    }
  });

  test('buildIdMap', () => {
    assert.deepEqual(buildIdMap(['9', '12:3', '4']), { '9': '1', '12:3': '2', '4': '3' });
    assert.deepEqual(buildIdMap([]), {});
  });

  test('node without inputs is tolerated', () => {
    const input = { '1': { class_type: 'A' }, '2': { class_type: 'B', inputs: { x: ['1', 0] } } };
    const { prompt } = normalizePrompt(input, { mode: 'topological' });
    assert.deepEqual(Object.keys(prompt), ['1', '2']);
    assert.deepEqual(prompt['2'].inputs.x, ['1', 0]);
  });
});

// ---------------------------------------------------------------------------
// 13. Cycle safety
// ---------------------------------------------------------------------------

describe('cycle safety', () => {
  for (const mode of MODES) {
    test(`all nodes survive a cycle (${mode})`, () => {
      const input = cyclic();
      const a = normalizePrompt(input, { mode });
      const b = normalizePrompt(cyclic(), { mode });
      assert.deepEqual(Object.keys(a.prompt), ['1', '2', '3']);
      assert.deepEqual(a.order.slice().sort(), ['10', '20', '5']);
      assert.deepEqual(a, b);
      assertIntegrity(input, a);
      // Cycle links still resolve to each other.
      const { idMap, prompt } = a;
      assert.deepEqual(prompt[idMap['10']].inputs.a, [idMap['20'], 0]);
      assert.deepEqual(prompt[idMap['20']].inputs.a, [idMap['10'], 0]);
    });
  }

  test('topological: independent node first, cycle members in compareIds order', () => {
    assert.deepEqual(topologicalOrder(cyclic()), ['5', '10', '20']);
  });
});
