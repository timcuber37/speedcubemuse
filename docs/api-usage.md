# API usage and cost review

The app records usage for each Claude and Voyage operation. Run the report
against the existing WCA MySQL/TiDB database to compare spending across the
website, Discord bot, and regulation ingestion jobs:

```bash
python scripts/usage_report.py --days 30
python scripts/usage_report.py --days 7 --group-by operation,model,source
python scripts/usage_report.py --since 2026-09-01 --until 2026-10-01 --group-by day
python scripts/usage_report.py --days 30 --source web --format csv > usage.csv
python scripts/usage_report.py --days 30 --format json > usage.json
```

Use the project's virtual environment and the same `DB_*` settings as the app.
Dates are UTC; `--since` is inclusive and `--until` is exclusive. The default is
the last 30 days. Reports are sorted by known estimated spend, descending.
The command only reads the database. There is no public billing endpoint.

## Reading the report

- **Calls:** SDK calls, including failures; excludes application cache hits.
- **Result hits:** valid game answers or declines served from the local/shared
  question cache, avoiding a model call altogether.
- **Est. USD:** sum of recorded estimates. An asterisk means some calls have
  unknown costs, so the total is incomplete. All-unknown groups say `unknown`.
- **Avg/call:** estimated USD per SDK call, excluding result hits. Left blank
  when any call's price is unknown, to avoid understating the average.
- **Cache read %:** cached input tokens divided by total input tokens, including
  cache writes. This is a token share, not the percentage of requests cached.
- **Errors:** SDK calls that raised an exception. Application parsing or SQL
  errors after a successful API response do not change its billing outcome.
- **Unpriced:** calls without usable token counts, a known model price, or a
  supported service tier. These are not treated as free.

CSV/JSON additionally include uncached input, output, cache-read, cache-write
tokens (including separate 5-minute and 1-hour writes), result-cache hit rate,
and average SDK call duration. The raw records include the requested and served
models, a random request ID shared by the stages of one user action, the
provider request ID when available, and the price version used.

Output tokens include any thinking tokens billed by the provider. The configured
`max_tokens` limit is not used as a proxy for actual consumption.

## Collection and persistence

After deploying this code, tracking is on by default:

```env
API_USAGE_ENABLED=true
API_USAGE_PERSIST=true
```

The first recorded event starts a worker that creates `api_usage_events` in the
existing WCA database. That database account needs CREATE, INSERT, and UPDATE
permissions for the table. Web workers and the bot share this table, so reports
survive Fly machine restarts. Recording does not change models or answers.

Events are logged immediately with an `api_usage` marker. A bounded background
queue batches database writes; database latency does not delay an answer. Failed
writes are retried once using the event ID to avoid duplicates. Queue overflow,
database outages, or forced process termination can leave gaps in the database
report. Warnings identify storage failures, and the corresponding events remain
in application logs subject to the hosting platform's log retention. Graceful
shutdown attempts a flush for up to five seconds. Export logs to review them:

```bash
python scripts/usage_report.py --logs web.log bot.log --days 30
```

Log input accepts plain JSON event lines and the text logs emitted by the app,
including Fly prefixes; repeated event IDs across files are counted once. This
reads the supplied files instead of the database. Keep log exports for the
period you want to audit. Database records currently have no automatic expiry.

Set `API_USAGE_PERSIST=false` for logs only (also useful for tests), or
`API_USAGE_ENABLED=false` to disable metering entirely. The usage records do not
contain questions, answers, SQL, API credentials, or user identifiers. Existing
application logging elsewhere is unchanged.

## Estimates versus actual credit usage

Tracking starts when this version is deployed; it cannot reconstruct historical
per-operation spending from the previous, incomplete cache logs. It also cannot
see requests made by other apps using the same provider key, or resolve billing
for failed/time-out requests and retries hidden inside the SDK.

For actual historical spending, credit balances, and discounts, use the
[Claude Console cost and usage reports](https://support.claude.com/en/articles/9534590-cost-and-usage-reporting-in-the-claude-console)
and [Voyage dashboard](https://dashboard.voyageai.com/). Reconcile the same UTC
period and API keys. Anthropic's
[Usage and Cost API](https://platform.claude.com/docs/en/manage-claude/usage-cost-api)
also provides organization reports with suitable admin credentials. None are
required for the app's own metering.

Public standard API rates checked on **2026-09-22**, in USD per million tokens:

| Model used by this app | Input | Output | 5m cache write | 1h cache write | Cache read |
|---|---:|---:|---:|---:|---:|
| Claude Sonnet 5 | 2 | 10 | 2.50 | 4 | 0.20 |
| Claude Opus 4.8 | 5 | 25 | 6.25 | 10 | 0.50 |
| Claude Haiku 4.5 | 1 | 5 | 1.25 | 2 | 0.10 |
| Voyage `voyage-3-large` | 0.18 | — | — | — | — |
| Voyage `rerank-2-lite` | 0.02 | — | — | — | — |

Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
and [Voyage pricing](https://docs.voyageai.com/docs/pricing). Reranking uses the
provider's processed-token count, including the query repeated per document.
These are list-price estimates before account credits, free allowances,
discounts, taxes, and negotiated pricing. Rates and `PRICING_VERSION` live in
`services/api_usage.py`; update both when changing pricing. Stored estimates
retain their original price version. Unknown models remain unpriced rather
than silently inheriting another model's rate.

## Initial audit and optimization priorities

This is a code-based assessment, not a ranking of measured production spend.
Total spend depends on both cost per call and traffic; collect a representative
week before deciding which feature to optimize first.

| Operation | When it runs | Cost driver / opportunity |
|---|---|---|
| `stats.generate_sql` | Every stats question, web or Discord | Schema prompt plus SQL output. Cache validated SQL for repeated questions, keyed by model and schema version, then execute it against current data. |
| `stats.repair_sql` | Once after a database execution error | Another generation with the failed SQL and error. Measure repair frequency and fix recurring schema/prompt issues. |
| `stats.summarize` | Web queries with successful, nonempty results | A separate call using the chosen SQL model, including Opus. Evaluate Haiku for this short step or template common answer shapes. |
| `delegate.answer` | Every Delegate question | Retrieved regulations, expanded parents/guidelines, history, and output. Evaluate thinking disabled for simple questions; enforce a source-token budget while preserving required citations. |
| `delegate.rewrite` | Delegate questions with history | Haiku sees history again before the main answer. Evaluate using only the context needed to resolve a follow-up. |
| `delegate.embed` | Every Delegate question | One Voyage query embedding. Cache embeddings/retrieval results by query, model, and regulations revision. |
| `delegate.rerank` | When vector search returns documents | Reranks up to 20 candidates by default. Lowering the final result count alone does not reduce tokens processed across all candidates. |
| `game.parse_question` | Typed game question absent from the valid result cache | Haiku with a 150-token output cap. Measure result-cache hits; concurrent misses for the same question can still make duplicate calls. |
| `regulations.embed` | Regulations ingestion | Voyage document embedding batches. Consider embedding only changed chunks if ingestion becomes material. |

The app's automatic guessing, candidate scoring, and name search make no paid
model calls. Database/hosting costs are outside this report.

Prompt caching deserves verification: both SQL and game prompts request it, but
Haiku 4.5 requires at least **4,096** tokens in the cached prefix, whereas Sonnet
5 and Opus 4.8 require **1,024**. A short prefix silently runs uncached. Check the
actual token and cache-write/read counts instead of relying on source comments.
Changing Delegate sources also prevents reuse of an identical cached prefix;
move stable instructions first if adding caching, then validate its economics.
See [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

Also review volume controls if usage spikes: Discord stats queries currently
have no command cooldown, and web limits use per-worker memory that resets on
restart. Durable user/day limits can bound spend more predictably.

Evaluate model/prompt changes on representative questions and correctness,
especially SQL repair and regulation citations. This change establishes the
measurements; it does not automatically apply those tradeoffs.
