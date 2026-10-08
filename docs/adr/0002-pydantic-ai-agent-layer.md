# ADR-0002: Pydantic AI as the agent layer

## Status
Accepted (2026-10-07).

## Context
The blueprint's default is plain code plus structured model calls; a framework is allowed when it brings a measurable advantage, recorded in an ADR (ARC-AGT-001).

## Options considered
- Hand-written tool loop on the Anthropic SDK.
- LangGraph: brings its own state and persistence, duplicating Temporal.
- CrewAI or AutoGen: multi-agent role play, not needed; AutoGen has been in maintenance since October 2025.
- Pydantic AI.

## Decision
Pydantic AI for the classifier and the responder: typed tools and outputs, usage limits, deterministic test models, and official Temporal durability (model requests and tool calls become activities).

## Consequences
- Agents are built Temporal-ready: a name, a model at construction, Pydantic deps, toolsets with explicit ids.
- Safety rules, validation and domain logic stay outside the framework, in plain Python.

## Exit plan
The framework touches only `frontdesk.agent`; tools call domain functions, so replacing it means rewriting two agent factories and the toolset.
