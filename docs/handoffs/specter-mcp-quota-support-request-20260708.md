# Specter MCP Quota Support Request

Date: 2026-07-08

## Context

Deal Intelligence, LeadGen, Slack, Codex, and ad hoc analysis can share the same Specter MCP workspace quota. A production LeadGen run hit:

`Daily MCP limit reached (250 tool calls/day). Resets at 00:00 UTC.`

The app can detect an exhausted quota only by making a small MCP probe before start, then stopping batches when Specter returns the quota error. It cannot know exact remaining MCP quota from Supabase because usage also happens outside Deal Intelligence.

## Request To Specter

Please provide one of the following for MCP usage:

- MCP quota/status endpoint with daily limit, remaining calls, and reset time.
- Quota headers or response metadata on MCP tool calls.
- MCP calls included in organization API logs.
- Higher daily MCP quota for the workspace.
- Separate production token/quota for Deal Intelligence.

## Current Public API Surface

Specter public API docs expose API rate/credit headers and organization API logs, but we do not have a confirmed MCP remaining-quota surface:

- Rate limits and credit headers: https://api.tryspecter.com/api-ref/rate_limits
- API call logs: https://api.tryspecter.com/api-ref/logs/list-api-call-logs
