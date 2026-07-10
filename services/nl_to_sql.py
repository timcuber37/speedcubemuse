"""Natural Language to SQL translation service."""
import json
import logging
import re
from anthropic import AsyncAnthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, QUERY_MODELS

logger = logging.getLogger(__name__)


def _execution_error(results) -> str:
    """Return the DB error message when results carry execute_query's error shape."""
    if isinstance(results, list) and len(results) == 1 \
            and isinstance(results[0], dict) and 'error' in results[0]:
        return str(results[0].get('message', 'Unknown error'))
    return None

# Structured output schema: the API guarantees the response is valid JSON in
# this shape, so no markdown-fence stripping is needed.
_SQL_OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": ["string", "null"],
                "description": "A single SQL SELECT query answering the question, or null if it cannot be answered",
            },
            "error": {
                "type": ["string", "null"],
                "description": "Brief reason the question cannot be answered with SQL, when sql is null",
            },
        },
        "required": ["sql", "error"],
        "additionalProperties": False,
    },
}


class NLToSQLService:
    """Service for translating natural language questions to SQL queries."""

    def __init__(self):
        """Initialize the NL-to-SQL service."""
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set. NL-to-SQL translation will not work.")
            self.client = None
        else:
            # Async client — the sync client blocks the event loop for the
            # duration of each API call (freezes the Discord bot for all users).
            self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        self.model = ANTHROPIC_MODEL

        # WCA database schema information for the AI
        self.schema_context = """
The World Cube Association (WCA) database contains the following tables:

**ranks_single** - World rankings for single solves
Columns: person_id (varchar), event_id (varchar), best (int, centiseconds), world_rank (int), continent_rank (int), country_rank (int)

**ranks_average** - World rankings for averages
Columns: person_id (varchar), event_id (varchar), best (int, centiseconds), world_rank (int), continent_rank (int), country_rank (int)

**results** - Competition results
Columns: id (bigint), competition_id (varchar), event_id (varchar), round_type_id (varchar), pos (int), best (int, centiseconds), average (int, centiseconds), person_name (varchar), person_id (varchar), person_country_id (varchar), format_id (varchar), regional_single_record (varchar), regional_average_record (varchar)

**persons** - Competitor information
Columns: wca_id (varchar), sub_id (int), name (varchar), country_id (varchar), gender (varchar)

**competitions** - Competition details
Columns: id (varchar), name (varchar), city_name (varchar), country_id (varchar), information (text), year (int), month (int), day (int), end_year (int), end_month (int), end_day (int), cancelled (int), event_specs (text), delegates (text), organizers (text), venue (varchar), venue_address (varchar), venue_details (varchar), external_website (varchar), cell_name (varchar), latitude_microdegrees (int), longitude_microdegrees (int)

**events** - Puzzle event information
Columns: id (varchar), name (varchar), rank (int), format (varchar)

**result_attempts** - Individual attempt values for each result (replaces value1-5 columns)
Columns: result_id (bigint, FK → results.id), attempt_number (tinyint), value (int, centiseconds; -1=DNF, -2=DNS, 0=no result)

**countries** - Country information
Columns: id (varchar), name (varchar), continent_id (varchar), iso2 (varchar)

Common event IDs:
- '333' = 3x3x3 Cube
- '222' = 2x2x2 Cube
- '444' = 4x4x4 Cube
- '555' = 5x5x5 Cube
- '666' = 6x6x6 Cube
- '777' = 7x7x7 Cube
- '333bf' = 3x3x3 Blindfolded
- '333fm' = 3x3x3 Fewest Moves
- '333oh' = 3x3x3 One-Handed
- 'clock' = Clock
- 'minx' = Megaminx
- 'pyram' = Pyraminx
- 'skewb' = Skewb
- 'sq1' = Square-1
- '444bf' = 4x4x4 Blindfolded
- '555bf' = 5x5x5 Blindfolded
- '333mbf' = 3x3x3 Multi-Blind

IMPORTANT NOTES:
- Time values are stored in centiseconds (1/100th of a second). Example: 1000 = 10.00 seconds, 6000 = 1:00.00
- EXCEPTIONS to centiseconds: for '333fm' (Fewest Moves), single values are move counts (e.g. 16 = 16 moves) and average values are move counts x 100 (e.g. 1900 = 19.00 moves). For '333mbf' (Multi-Blind), the value encodes the whole result as DDTTTTTMM (DD = 99 minus points, TTTTT = time in seconds, MM = missed cubes); lower values are better, so ordering still works, but do not treat it as a time. Always include event_id in the SELECT when querying '333fm' or '333mbf' so values can be displayed correctly
- -1 means DNF (Did Not Finish), -2 means DNS (Did Not Start), 0 means no result
- The 'best' column in ranks_single and ranks_average contains the person's best time
- World rank 1 means world record holder
- RECORDS: results.regional_single_record and results.regional_average_record mark records set by that result: 'WR' = world record, 'NR' = national record, continental records are 'ER' (Europe), 'NAR' (North America), 'SAR' (South America), 'AsR' (Asia), 'AfR' (Africa), 'OcR' (Oceania). Use these columns for questions about records set or held
- NAMES: person names may include a parenthesized local-script form (e.g. "Ken'ichi Ueno (上野健一)"). Always match names fuzzily: WHERE p.name LIKE '%Max Park%'
- COUNTRIES: country_id values are English country names ('China', 'Germany', 'United Kingdom'), except the United States which is 'USA'. When unsure of the exact id, join the countries table and match countries.name or countries.iso2
- DATES: competitions store dates as integer columns year/month/day (start) and end_year/end_month/end_day — there is no DATE column. cancelled = 1 marks cancelled competitions; exclude them by default
- ROUNDS: round_type_id 'f' = Final and 'c' = Combined Final (treat both as finals); '1'/'2'/'3' = numbered rounds, 'd'/'e'/'g' = combined rounds, '0'/'h' = qualification, 'b' = B final
- Individual solve attempt values are in result_attempts, NOT in the results table. Join on result_attempts.result_id = results.id
- For questions about overall best/average in a round, use the results table. For individual attempt values, join result_attempts
- Unless the question implies otherwise, add LIMIT 50 to queries that can return many rows

EXAMPLE QUERIES:

World record for 3x3 single:
SELECT p.name, r.best, r.world_rank, p.country_id
FROM ranks_single r
JOIN persons p ON r.person_id = p.wca_id
WHERE r.event_id = '333' AND r.world_rank = 1

World record for 3x3 average:
SELECT p.name, r.best, r.world_rank, p.country_id
FROM ranks_average r
JOIN persons p ON r.person_id = p.wca_id
WHERE r.event_id = '333' AND r.world_rank = 1

Top 10 fastest 3x3 singles:
SELECT p.name, r.best, r.world_rank, p.country_id
FROM ranks_single r
JOIN persons p ON r.person_id = p.wca_id
WHERE r.event_id = '333'
ORDER BY r.world_rank ASC
LIMIT 10

Competition results for a specific event:
SELECT person_name, best, average, pos, competition_id
FROM results
WHERE event_id = '333' AND best > 0
ORDER BY best ASC
LIMIT 20

Person with most competition results:
SELECT person_name, person_id, COUNT(*) as result_count
FROM results
GROUP BY person_id, person_name
ORDER BY result_count DESC
LIMIT 10

All individual attempts for a specific result (using result_attempts):
SELECT r.person_name, r.competition_id, ra.attempt_number, ra.value
FROM results r
JOIN result_attempts ra ON ra.result_id = r.id
WHERE r.event_id = '333' AND r.person_id = 'WCAID'
ORDER BY ra.attempt_number

Person with the most world record singles:
SELECT person_name, COUNT(*) as wr_count
FROM results
WHERE regional_single_record = 'WR'
GROUP BY person_id, person_name
ORDER BY wr_count DESC
LIMIT 10

A specific person's best 3x3 time (fuzzy name match):
SELECT p.name, r.best, r.world_rank, p.country_id
FROM ranks_single r
JOIN persons p ON r.person_id = p.wca_id
WHERE p.name LIKE '%Max Park%' AND r.event_id = '333'

Competitions held in the United States in 2025:
SELECT name, city_name, year, month, day
FROM competitions
WHERE country_id = 'USA' AND year = 2025 AND cancelled = 0
ORDER BY month, day
LIMIT 50

Winner of the 3x3 final at the 2023 World Championship:
SELECT r.person_name, r.best, r.average, r.pos
FROM results r
JOIN competitions c ON r.competition_id = c.id
WHERE c.name LIKE '%World Championship 2023%' AND r.event_id = '333'
  AND r.round_type_id IN ('f', 'c') AND r.pos = 1
"""

        # Static across all requests — built once so the cached prefix is byte-identical.
        self.system_prompt = f"""You are a SQL expert for the World Cube Association (WCA) database.

{self.schema_context}

Generate a SQL query that answers the user's question: set "sql" to the query and "error" to null.
If the question cannot be answered with SQL against this schema, set "sql" to null and "error" to a brief reason."""


    def _resolve_model(self, model_key: str) -> str:
        """Map a user-supplied model key to an allowed model ID.

        Falls back to the configured default model for unknown or missing keys,
        so callers can never request an arbitrary model string.
        """
        if model_key and model_key in QUERY_MODELS:
            return QUERY_MODELS[model_key]
        return self.model

    async def _create_message(self, model_id: str, max_tokens: int, system, messages: list, output_config: dict = None):
        """Call the Anthropic API asynchronously."""
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            # Sonnet 5 runs adaptive thinking when this field is omitted, and
            # thinking tokens count against max_tokens — keep it off for latency.
            "thinking": {"type": "disabled"},
        }
        if output_config:
            kwargs["output_config"] = output_config
        response = await self.client.messages.create(**kwargs)
        usage = response.usage
        logger.info(
            f"NL-to-SQL model: requested={model_id}, served={response.model}, "
            f"cache_read={usage.cache_read_input_tokens}, cache_write={usage.cache_creation_input_tokens}"
        )
        return response

    async def _generate_sql(self, user_content: str, model: str = None) -> str:
        """Run one structured-output SQL generation call; returns validated SQL or None."""
        response = await self._create_message(
            self._resolve_model(model),
            max_tokens=1024,
            # The system prompt is identical on every request — cache it so
            # repeat requests read it at ~10% of the input price.
            system=[{
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": _SQL_OUTPUT_FORMAT},
        )

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)

        if not data.get("sql"):
            logger.warning(f"Translation declined: {data.get('error')}")
            return None

        sql_query = data["sql"].strip()

        if not self.validate_sql(sql_query):
            logger.warning(f"Rejected unsafe SQL: {sql_query}")
            return None

        return sql_query

    async def translate_to_sql(self, question: str, model: str = None) -> str:
        """
        Translate a natural language question to a SQL query.

        Args:
            question: The user's question in natural language
            model: Optional model key ("opus", "sonnet", "haiku"); falls back
                to the configured default when missing or unrecognized.

        Returns:
            SQL query string, or None if translation fails
        """
        if not self.client:
            logger.error("Anthropic client not initialized. Cannot translate to SQL.")
            return None

        try:
            return await self._generate_sql(f"User Question: {question}", model=model)
        except Exception as e:
            logger.error(f"Error translating to SQL: {e}")
            return None

    async def repair_sql(self, question: str, sql_query: str, db_error: str, model: str = None) -> str:
        """Ask the model to fix a SQL query that failed at the database."""
        if not self.client:
            return None

        try:
            return await self._generate_sql(
                f"""User Question: {question}

This SQL query failed:
{sql_query}

Database error:
{db_error}

Generate a corrected SQL query that answers the question.""",
                model=model,
            )
        except Exception as e:
            logger.error(f"Error repairing SQL: {e}")
            return None

    async def answer_question(self, question: str, execute, model: str = None):
        """
        Translate a question to SQL, execute it, and repair once on a database error.

        Args:
            question: The user's question in natural language
            execute: Async callable that runs a SQL string and returns result rows
                (execute_query's error shape [{"error": ..., "message": ...}] on failure)
            model: Optional model key ("opus", "sonnet", "haiku")

        Returns:
            (sql_query, results) — sql_query is None when translation fails,
            and results is None only in that case.
        """
        sql_query = await self.translate_to_sql(question, model=model)
        if not sql_query:
            return None, None

        results = await execute(sql_query)

        db_error = _execution_error(results)
        if db_error:
            logger.info(f"SQL failed, attempting repair: {db_error}")
            fixed_sql = await self.repair_sql(question, sql_query, db_error, model=model)
            if fixed_sql:
                fixed_results = await execute(fixed_sql)
                if _execution_error(fixed_results) is None:
                    return fixed_sql, fixed_results
                logger.warning("Repaired SQL also failed; returning original error")

        return sql_query, results

    async def summarize_results(self, question: str, results: list, model: str = None) -> str:
        """Return a 1-2 sentence natural language answer to the question given the query results."""
        # Don't summarize execution-error rows — the UI already shows the error.
        if not self.client or not results or _execution_error(results):
            return None

        # Cap rows sent to Claude to keep token usage low
        sample = results[:10]
        rows_text = '\n'.join(str(row) for row in sample)
        if len(results) > 10:
            rows_text += f'\n... and {len(results) - 10} more rows'

        try:
            response = await self._create_message(
                self._resolve_model(model),
                max_tokens=150,
                system="You are a helpful assistant summarizing WCA competition database query results. Write 1-2 sentences directly answering the user's question based on the data. Be concise and specific — include key names, numbers, or times from the results. Do not mention SQL.",
                messages=[{
                    "role": "user",
                    "content": f"Question: {question}\n\nResults:\n{rows_text}"
                }]
            )
            return next(b.text for b in response.content if b.type == "text").strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return None

    def validate_sql(self, sql_query: str) -> bool:
        """Validate that SQL is a safe read-only SELECT query."""
        sql_upper = sql_query.strip().upper()

        if not sql_upper.startswith('SELECT'):
            return False

        forbidden = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE',
            'CREATE', 'REPLACE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE'
        ]
        for keyword in forbidden:
            # Check as whole word to avoid false positives (e.g. "UPDATES" in a column name)
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False

        return True

