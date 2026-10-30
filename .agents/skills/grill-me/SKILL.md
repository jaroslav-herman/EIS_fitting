---
name: grill-me
description: For larger or ambiguous updates to the application, interview the user before implementing so the implementation matches their mental model. Use when a request spans multiple layers, changes a whole workflow, or the desired behavior is not fully specified; do not use for small, unambiguous fixes.
---

# Grill Me

Before writing code for a larger update, grill the user with targeted questions so the implementation and the user's point of view match. Ask early, ask concretely, and do not guess silently.

## When to grill

Grill when any of these hold:

- The request changes a whole workflow, adds a major feature, or touches several layers (state, services, persistence, GUI).
- The request is ambiguous about behavior, scope, or UX (what should happen when X, where does the control live, what is shown to the user).
- The change could reasonably be implemented in more than one way with different user-visible results.

Do not grill for small, unambiguous fixes (a typo, a one-line bug, a clearly specified tweak). One short clarifying question is fine there.

## How to grill

- Ask about the user's intent and mental model, not implementation trivia: "How do you imagine this working day to day?", "What should the user see when ...?", "Should this be automatic or require an explicit action?"
- Ask 2-4 focused questions at a time, each with concrete options when the options are known; leave room for free text.
- Cover: scope (what is in and out), user-facing behavior (controls, dialogs, plots, status text), state/persistence impact (what gets saved, when it is invalidated), and edge cases (empty selection, batch runs, unsaved changes).
- Wait for the answers before editing. If an answer is unclear, ask one follow-up rather than assuming.
- Once the answers are clear, summarize the agreed behavior in one short paragraph, confirm it matches, then implement.

## After the answers

- Implement the smallest change that matches the agreed behavior.
- If during implementation a decision materially diverges from what was agreed, stop and re-ask instead of improvising.
