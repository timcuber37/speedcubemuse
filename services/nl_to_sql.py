"""Natural Language to SQL translation service."""
import logging
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

logger = logging.getLogger(__name__)


class NLToSQLService:
    """Service for translating natural language questions to SQL queries."""

    def __init__(self):
        """Initialize the NL-to-SQL service."""
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set. NL-to-SQL translation will not work.")
            self.client = None
        else:
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

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
- -1 means DNF (Did Not Finish), -2 means DNS (Did Not Start), 0 means no result
- The 'best' column in ranks_single and ranks_average contains the person's best time
- World rank 1 means world record holder
- Individual solve attempt values are in result_attempts, NOT in the results table. Join on result_attempts.result_id = results.id
- For questions about overall best/average in a round, use the results table. For individual attempt values, join result_attempts

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
"""
    
    async def translate_to_sql(self, question: str) -> str:
        """
        Translate a natural language question to a SQL query.
        
        Args:
            question: The user's question in natural language
            
        Returns:
            SQL query string, or None if translation fails
        """
        if not self.client:
            logger.error("Anthropic client not initialized. Cannot translate to SQL.")
            return None

        try:
            system_prompt = f"""You are a SQL expert for the World Cube Association (WCA) database.

{self.schema_context}

Generate SQL queries based on natural language questions. Return ONLY the SQL query, no explanations or markdown formatting.
If the question cannot be answered with SQL, return "ERROR: Cannot be answered with SQL"."""

            user_prompt = f"""User Question: {question}

Generate a SQL query that answers this question. Return ONLY the SQL query, no explanations.

SQL Query:"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            sql_query = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith("```"):
                sql_query = sql_query.split("```")[1]
                if sql_query.startswith("sql"):
                    sql_query = sql_query[3:]
                sql_query = sql_query.strip()
            
            # Check for error
            if sql_query.startswith("ERROR:"):
                logger.warning(f"Translation error: {sql_query}")
                return None
            
            if not self.validate_sql(sql_query):
                logger.warning(f"Rejected unsafe SQL: {sql_query}")
                return None

            return sql_query

        except Exception as e:
            logger.error(f"Error translating to SQL: {e}")
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
            import re
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False

        return True

