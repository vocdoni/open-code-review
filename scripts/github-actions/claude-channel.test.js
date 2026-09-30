#!/usr/bin/env node

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "../..");
const workflow = fs.readFileSync(path.join(root, ".github/workflows/claude-review-channel.yml"), "utf8");
const worker = fs.readFileSync(path.join(root, ".github/workflows/claude-review.yml"), "utf8");
const repository = "vocdoni/open-code-review";
const defaults = {
  runner_labels: '["ubuntu-latest"]',
  model: "",
  gateway_url: "",
  claude_version: "2.1.285",
  effort: "high",
  review_concurrency: "1",
  llm_timeout: "300",
  review_task_timeout: "30",
  max_tokens_budget: "1500000",
};

// Extract this workflow's block mappings without a YAML dependency. Keep nested
// bodies as text; reject duplicate keys and unsupported indentation rather than
// letting a matching comment or a different scope satisfy a contract.
function mapping(text, indent) {
  assert.strictEqual(typeof text, "string", "missing workflow mapping");
  const result = {};
  let key;
  for (const line of text.split("\n")) {
    if (!line.trim() || /^\s*#/.test(line)) continue;
    const match = /^( *)([A-Za-z_][A-Za-z0-9_-]*):(?: (.*))?$/.exec(line);
    if (match && match[1].length === indent) {
      key = match[2];
      assert(!Object.prototype.hasOwnProperty.call(result, key), `duplicate workflow key: ${key}`);
      result[key] = match[3] || "";
    } else {
      assert(key && line.startsWith(" ".repeat(indent + 2)), `unsupported workflow line: ${line}`);
      result[key] += `\n${line}`;
    }
  }
  return result;
}

function scalar(value) {
  const text = value.trim();
  if (text.startsWith("'") && text.endsWith("'")) return text.slice(1, -1).replace(/''/g, "'");
  if (text.startsWith('"') && text.endsWith('"')) return JSON.parse(text);
  return text;
}

function keys(object, expected) {
  assert.deepStrictEqual(Object.keys(object).sort(), [...expected].sort());
}

const channel = mapping(workflow, 0);
const triggers = mapping(channel.on, 2);
const call = mapping(triggers.workflow_call, 4);
const inputs = mapping(call.inputs, 6);
const jobs = mapping(channel.jobs, 2);
const review = mapping(jobs.review, 4);
const forwarded = mapping(review.with, 6);

function testCentralImmutablePins() {
  const pin = review.uses.match(/^([^/]+\/[^/]+)\/\.github\/workflows\/claude-review\.yml@([a-f0-9]{40})$/);
  assert(pin, "the worker must be called at a full immutable commit SHA");
  assert.strictEqual(pin[1], repository);
  assert.strictEqual(forwarded.source_repository, repository);
  assert.strictEqual(forwarded.source_sha, pin[2], "worker and source must use the same central pin");
}

function testChannelOnlyDelegatesToWorker() {
  keys(channel, ["name", "on", "permissions", "jobs"]);
  keys(triggers, ["workflow_call"]);
  keys(call, ["inputs", "secrets", "outputs"]);
  keys(jobs, ["review"]);
  keys(review, ["if", "uses", "with", "secrets"]);
  // In particular, there are no steps, checkout, env, or caller-controlled source pins.
  keys(inputs, Object.keys(defaults));
  keys(forwarded, ["source_repository", "source_sha", ...Object.keys(defaults)]);
}

function testPublicInputDefaultsMatchWorker() {
  const workerCall = mapping(mapping(mapping(worker, 0).on, 2).workflow_call, 4);
  const workerInputs = mapping(workerCall.inputs, 6);
  for (const [name, expected] of Object.entries(defaults)) {
    const definition = mapping(inputs[name], 8);
    const workerDefinition = mapping(workerInputs[name], 8);
    keys(definition, ["description", "type", "default"]);
    assert.strictEqual(definition.type, "string", `${name} must remain a string input`);
    assert.strictEqual(scalar(definition.default), expected, `${name} default changed`);
    assert.strictEqual(definition.type, workerDefinition.type);
    assert.strictEqual(scalar(definition.default), scalar(workerDefinition.default), `${name} differs from the worker`);
  }
}

function testAllPublicInputsForwardWithoutFallbacks() {
  for (const name of Object.keys(defaults)) {
    assert.strictEqual(forwarded[name], `\${{ inputs.${name} }}`, `${name} must be forwarded unchanged`);
  }
  assert(!/\bvars\./.test(workflow), "consumer variable and model fallbacks belong to the worker");
}

function testTrustedEventGuardAndPermissions() {
  assert.deepStrictEqual(mapping(channel.permissions, 2), {
    contents: "read",
    "pull-requests": "write",
  });
  assert(review.if.startsWith(">-\n"), "expected a folded job guard");
  const guard = review.if.slice(3).trim().replace(/\s+/g, " ");
  assert.strictEqual(guard, [
    "github.event_name == 'pull_request_target'",
    "!github.event.pull_request.draft",
    "github.event.pull_request.head.repo.full_name == github.repository",
    "github.event.pull_request.user.type != 'Bot'",
  ].join(" && "), "the worker call must reject other events, drafts, forks, and bots");
}

function testExplicitSecretScopeAndOutputForwarding() {
  const secrets = mapping(call.secrets, 6);
  keys(secrets, ["gateway_token"]);
  const token = mapping(secrets.gateway_token, 8);
  keys(token, ["description", "required"]);
  assert.strictEqual(token.required, "true");
  assert.deepStrictEqual(mapping(review.secrets, 6), {
    gateway_token: "${{ secrets.gateway_token }}",
  });
  assert.strictEqual(workflow.split("${{ secrets.gateway_token }}").length - 1, 1);
  assert(!/secrets:\s*inherit/.test(workflow), "only the explicit gateway token may be forwarded");

  const outputs = mapping(call.outputs, 6);
  keys(outputs, ["summary_comment_url"]);
  const summary = mapping(outputs.summary_comment_url, 8);
  keys(summary, ["description", "value"]);
  assert.strictEqual(summary.value, "${{ jobs.review.outputs.summary_comment_url }}");
}

const tests = [
  testCentralImmutablePins,
  testChannelOnlyDelegatesToWorker,
  testPublicInputDefaultsMatchWorker,
  testAllPublicInputsForwardWithoutFallbacks,
  testTrustedEventGuardAndPermissions,
  testExplicitSecretScopeAndOutputForwarding,
];
for (const test of tests) {
  test();
  console.log(`PASS ${test.name}`);
}
console.log(`All ${tests.length} Claude channel tests passed.`);
