// Run with: node --test tests/test_demo.mjs
// Tests the actual embedded script using a minimal DOM; no browser dependencies.
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const html = readFileSync('docs/index.html', 'utf8');
test('all local image assets exist', () => {
  for (const [, path] of html.matchAll(/<img src="([^"]+)"/g)) {
    assert.ok(existsSync(`docs/${path}`), path);
  }
  assert.ok(!html.includes('RESULT_TEXT'));
});
test('intervention slider updates contrast and reset restores default', () => {
  const elements = new Map();
  const document = {querySelector(id) {
    if (!elements.has(id)) elements.set(id, {
      value: '8', textContent: '', attrs: {}, handlers: {},
      setAttribute(k, v) { this.attrs[k] = v; },
      addEventListener(k, fn) { this.handlers[k] = fn; },
    });
    return elements.get(id);
  }};
  runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], { document });
  assert.equal(elements.get('#contrast').textContent, '-8.0');
  const slider = elements.get('#effect');
  slider.value = '0'; slider.handlers.input();
  assert.equal(elements.get('#yes').attrs.d, elements.get('#no').attrs.d);
  slider.value = '16'; slider.handlers.input();
  assert.equal(elements.get('#contrast').textContent, '-16.0');
  elements.get('#reset').handlers.click();
  assert.equal(elements.get('#contrast').textContent, '-8.0');
});
