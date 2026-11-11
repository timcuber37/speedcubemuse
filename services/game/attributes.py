"""The attribute schema — single source of truth for the whole game.

Three consumers read this module and they must never drift apart:

    scripts/build_cuber_profiles.py  computes these values into `cuber_profiles`
    services/game/engine.py          turns them into questions and filters on them
    services/game/question_parser.py renders them as the vocabulary the model maps onto

Every question the game can ever ask is a *binary* predicate over one attribute,
because a human can only answer yes/no. Non-boolean attributes therefore expand
into several binary questions:

    bool         -> one question       ("has_wr is true?")
    categorical  -> one per value      ("continent_id == _Europe?")
    numeric      -> one per threshold  ("comp_count >= 50?")
    multi        -> one per value held, plus one per event group
                    ("top100_events contains 444?",
                     "top100_events overlaps the big cubes?")

Numerics are stored raw rather than pre-bucketed. Bucketing at build time throws
away information the engine could have used, and thresholds are cheaper to retune
here than to recompute across 6.7M rows.

Two things in here are easy to get wrong and are called out where they are
defined: record attributes are *cumulative* (a world record also satisfies
"continental record or better"), and "ever set" is a different question from
"currently holds".
"""
from __future__ import annotations

from dataclasses import dataclass

# Event groupings speedcubers actually talk in. Nobody asks "is their best event
# 666?" — they ask "are they a big cube person?".
EVENT_GROUPS: dict[str, tuple[str, ...]] = {
    'big_cubes':    ('444', '555', '666', '777'),
    'blind_events': ('333bf', '444bf', '555bf', '333mbf'),
    'side_events':  ('pyram', 'minx', 'skewb', 'sq1', 'clock'),
}

# The official WCA events the game treats as current: what "competed in every
# current event" checks, and the only events that count for current rankings.
#
# Declared rather than read from the export. The export already lists FTO as
# a live event (its rank is below the 900 that marks retired ones), but it isn't
# official, and requiring it would make "every current event" impossible. Add
# 'fto' here when it becomes official. A listed event the export marks retired
# still drops out on its own.
CURRENT_EVENTS: tuple[str, ...] = (
    '222', '333', '444', '555', '666', '777',
    '333bf', '444bf', '555bf', '333mbf',
    '333oh', '333fm', 'clock', 'minx', 'pyram', 'skewb', 'sq1',
)

# How much attention an event actually gets, used to weight the fame score that
# sorts difficulty tiers. A 3x3 world record makes the rounds; a 5BLD one is
# known inside its own community. Tier 1 is the WCA's headline event, tier 2 the
# popular side events, tier 3 the less-followed ones, tier 4 the events people
# largely chase personal bests in.
EVENT_TIERS: dict[str, int] = {
    '333': 1,
    '222': 2, '444': 2, 'pyram': 2, '333oh': 2, 'skewb': 2,
    'clock': 3, '555': 3, 'sq1': 3, 'minx': 3, '666': 3, '777': 3, '333bf': 3,
    '444bf': 4, '555bf': 4, '333fm': 4, '333mbf': 4,
}

# Multiplier per tier. The 4:1 spread between 3x3 and the PB events is a
# judgment call and the single number to change if the ranking feels off.
EVENT_TIER_WEIGHT: dict[int, float] = {1: 1.0, 2: 0.65, 3: 0.40, 4: 0.25}

# Retired events (Magic, Master Magic, With Feet, old-style Multi-Blind) and
# anything the WCA adds later fall here until listed above.
DEFAULT_EVENT_TIER = 4


def event_weight(event_id: str) -> float:
    """Popularity multiplier for an achievement in this event."""
    return EVENT_TIER_WEIGHT[EVENT_TIERS.get(event_id, DEFAULT_EVENT_TIER)]


GROUP_LABELS = {
    'big_cubes': 'a big cube event (4x4 and up)',
    'blind_events': 'a blindfolded event',
    'side_events': 'a side event (Pyraminx, Megaminx, Skewb, Square-1, Clock)',
}

# WCA event ids written the way a cuber would say them.
EVENT_NAMES = {
    '222': '2x2', '333': '3x3', '444': '4x4', '555': '5x5',
    '666': '6x6', '777': '7x7',
    '333bf': '3x3 Blindfolded', '444bf': '4x4 Blindfolded',
    '555bf': '5x5 Blindfolded', '333mbf': 'Multi-Blind',
    '333fm': 'Fewest Moves', '333oh': '3x3 One-Handed',
    'clock': 'Clock', 'minx': 'Megaminx', 'pyram': 'Pyraminx',
    'skewb': 'Skewb', 'sq1': 'Square-1',
    # Events the WCA has retired. People still hold results in them, so they
    # reach the game through `main_event` — left out, they render as raw ids and
    # the player is asked "is their strongest event mmagic?".
    '333ft': '3x3 With Feet', 'magic': 'Magic', 'mmagic': 'Master Magic',
    '333mbo': 'Multi-Blind (old style)',
    # Not yet a full WCA event, but present in the export.
    'fto': 'Face-Turning Octahedron',
}


def event_label(event_id: str) -> str:
    return EVENT_NAMES.get(event_id, str(event_id))


def country_label(country_id: str) -> str:
    """WCA country ids substitute '_' for an apostrophe.

    'Cote d_Ivoire' and 'Democratic People_s Republic of Korea' are the two in
    the export. Left raw they reach players in questions *and* on the result
    card, so this is shared rather than living in the question renderer.
    """
    return (country_id or '').replace('_', "'")


def group_of(event_id: str) -> str | None:
    """The group an event belongs to, or None for 2x2/3x3/OH/FMC."""
    for group, members in EVENT_GROUPS.items():
        if event_id in members:
            return group
    return None


@dataclass(frozen=True)
class Attribute:
    """One precomputed fact about a cuber.

    `question` is phrased for Mode 2, where the app is asking a human about the
    cuber they're thinking of, so it reads as a natural yes/no question. For
    numeric, categorical and multi attributes it contains a `{value}`
    placeholder.

    `aliases` feed the question parser's vocabulary — they are the phrasings a
    player is likely to type for this attribute, and exist to make the mapping
    from free text unambiguous rather than to be matched literally.
    """

    key: str
    kind: str  # 'bool' | 'categorical' | 'numeric' | 'multi'
    question: str
    aliases: tuple[str, ...] = ()
    thresholds: tuple[int, ...] = ()  # numeric only
    values: tuple[str, ...] = ()  # categorical only; may be filled at build time
    # Categorical attributes whose value set is discovered from the data rather
    # than declared here (country, main event). The builder fills `values`.
    dynamic_values: bool = False
    # Multi attributes whose values are WCA event ids, so the engine can also
    # ask about whole event groups.
    event_valued: bool = False
    # Boolean attributes that are necessarily true whenever this one is. A world
    # record is also a continental record, so answering yes to `has_wr` settles
    # `has_cr_or_better` and the engine must not ask it. Applied transitively,
    # and in reverse for a "no". Relations that span other kinds of attribute
    # are declared in RELATIONS below.
    implies: tuple[str, ...] = ()
    # Alternative views of the same topic. Once a game asks about one attribute
    # in a family, use that attribute for any refinements instead of switching
    # to another phrasing. These are overlaps, not logical equivalences: career
    # length and debut year differ for a retired cuber, for example.
    question_family: str | None = None


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

_IDENTITY = [
    Attribute(
        key='gender',
        kind='categorical',
        question='Is the cuber {value}?',
        aliases=('gender', 'male', 'female', 'man', 'woman', 'girl', 'guy'),
        values=('m', 'f', 'o'),
    ),
    Attribute(
        key='continent_id',
        kind='categorical',
        question='Are they from {value}?',
        aliases=('continent', 'europe', 'asia', 'north america', 'africa',
                 'oceania', 'south america'),
        values=('_Africa', '_Asia', '_Europe', '_North America',
                '_Oceania', '_South America'),
    ),
    Attribute(
        key='country_id',
        kind='categorical',
        question='Do they represent {value}?',
        aliases=('country', 'nationality', 'from', 'represent'),
        dynamic_values=True,
    ),
]

# ---------------------------------------------------------------------------
# Era — when they competed
# ---------------------------------------------------------------------------

_ERA = [
    Attribute(
        key='first_year',
        kind='numeric',
        question='Did they start competing in {value} or later?',
        aliases=('first competition', 'started competing', 'debut', 'rookie year'),
        thresholds=(2010, 2013, 2016, 2019, 2022),
        question_family='career_start',
    ),
    Attribute(
        key='last_year',
        kind='numeric',
        question='Have they competed in {value} or later?',
        aliases=('last competition', 'most recent competition', 'still competing'),
        thresholds=(2012, 2017, 2021, 2024),
        question_family='recent_activity',
    ),
    Attribute(
        key='years_active',
        kind='numeric',
        question='Have they been competing for at least {value} years?',
        aliases=('years active', 'career length', 'how long competing'),
        thresholds=(5, 8, 12, 18),
        question_family='career_start',
    ),
    Attribute(
        key='is_active',
        kind='bool',
        question='Are they still actively competing?',
        aliases=('active', 'retired', 'still competing', 'currently competing'),
        question_family='recent_activity',
    ),
]

# ---------------------------------------------------------------------------
# Records — the headline achievements
# ---------------------------------------------------------------------------

_RECORDS = [
    # Records are a hierarchy: WR > CR > NR. A world record holder also satisfies
    # "continental record or better" and "national record or better", so these
    # three are cumulative rather than mutually exclusive. Treating them as
    # separate buckets is wrong in a way players notice immediately — the FMC
    # world record holder got a flat "no" to "continental record or better".
    Attribute(
        key='has_wr',
        kind='bool',
        question='Have they ever set a world record?',
        aliases=('world record', 'wr', 'set a world record'),
        implies=('has_cr_or_better',),
    ),
    Attribute(
        key='has_cr_or_better',
        kind='bool',
        question='Have they ever set a continental record or better?',
        aliases=('continental record', 'cr', 'european record', 'nar',
                 'asian record', 'continental record or better'),
        implies=('has_nr_or_better',),
    ),
    Attribute(
        key='has_nr_or_better',
        kind='bool',
        question='Have they ever set a national record or better?',
        aliases=('national record', 'nr', 'national record or better'),
    ),
    # "Ever set" and "currently holds" are different questions, and players ask
    # both. These come from the current rank tables, where the hierarchy is
    # automatic: rank 1 in the world is necessarily rank 1 in your continent and
    # your country too.
    Attribute(
        key='currently_wr',
        kind='bool',
        question='Do they currently hold a world record?',
        aliases=('currently hold a world record', 'current wr',
                 'world number one', 'ranked first in the world', 'reigning wr'),
        # Rank 1 in the world is also rank 1 in your continent, and top 10.
        implies=('currently_cr_or_better', 'top10_any'),
    ),
    Attribute(
        key='currently_cr_or_better',
        kind='bool',
        question='Do they currently hold a continental record or better?',
        aliases=('currently hold a continental record', 'current cr',
                 'best in their continent'),
        implies=('currently_nr_or_better',),
    ),
    Attribute(
        key='currently_nr_or_better',
        kind='bool',
        question='Do they currently hold a national record or better?',
        aliases=('currently hold a national record', 'current nr',
                 'best in their country', 'national champion time'),
    ),
    Attribute(
        key='wr_single_count',
        kind='numeric',
        question='Have they set a single-solve world record at least {value} times?',
        aliases=('world record singles', 'wr single count'),
        thresholds=(1, 2, 5),
    ),
    Attribute(
        key='wr_average_count',
        kind='numeric',
        question='Have they set a world record average at least {value} times?',
        aliases=('world record averages', 'wr average count'),
        thresholds=(1, 2, 5),
    ),
    Attribute(
        key='cr_count',
        kind='numeric',
        question='Have they set at least {value} continental records?',
        aliases=('continental record count',),
        thresholds=(1, 5, 20),
    ),
    Attribute(
        key='nr_count',
        kind='numeric',
        question='Have they set at least {value} national records?',
        aliases=('national record count',),
        thresholds=(1, 10, 50),
    ),
]

# ---------------------------------------------------------------------------
# Current rankings
# ---------------------------------------------------------------------------

_RANKINGS = [
    # Every event a player is world top-100 in, not just 3x3. Stored as a set so
    # one attribute answers "are they top 100 in 4x4?", "...in any blind event?"
    # and "...in 3x3?" instead of needing a boolean per event. Either the single
    # or average ranking qualifies; being in both still counts as one event.
    Attribute(
        key='top100_events',
        kind='multi',
        question='Are they currently in the world top 100 for {value} '
                 'in either the single or average rankings?',
        aliases=('top 100', 'ranked in', 'world ranking', 'top hundred'),
        event_valued=True,
    ),
    Attribute(
        key='top10_any',
        kind='bool',
        question='Are they currently in the world top 10 for any event '
                 'in either the single or average rankings?',
        aliases=('top 10', 'top ten', 'world top 10'),
    ),
    Attribute(
        key='top100_event_count',
        kind='numeric',
        question='Are they in the world top 100 for at least {value} events, '
                 'counting either single or average rankings?',
        aliases=('how many events top 100', 'top 100 events'),
        thresholds=(1, 2, 5),
    ),
    Attribute(
        key='main_event',
        kind='categorical',
        question='Is their strongest event {value}?',
        aliases=('main event', 'best event', 'specialty', 'strongest event'),
        dynamic_values=True,
    ),
    Attribute(
        key='main_event_group',
        kind='categorical',
        question='Is their strongest event {value}?',
        aliases=('big cubes', 'blind', 'side events', 'what kind of event',
                 'type of event'),
        values=('big_cubes', 'blind_events', 'side_events'),
    ),
]

# ---------------------------------------------------------------------------
# Event profile — what kind of cuber they are
# ---------------------------------------------------------------------------

_EVENT_PROFILE = [
    # These mirror EVENT_GROUPS and are derived from the same grouping in the
    # builder, so a change to a group can't leave the two disagreeing.
    Attribute(
        key='is_bld_specialist',
        kind='bool',
        question='Are they primarily a blindfolded solver?',
        aliases=('blindfolded', 'bld', 'blind', 'multiblind', 'mbld',
                 'blind events'),
    ),
    Attribute(
        key='is_bigcube_specialist',
        kind='bool',
        question='Are they primarily a big cube solver (4x4 and up)?',
        aliases=('big cubes', 'bigcube', '4x4', '5x5', '6x6', '7x7'),
    ),
    Attribute(
        key='is_sideevent_specialist',
        kind='bool',
        question='Are they primarily a side event solver '
                 '(Pyraminx, Megaminx, Skewb, Square-1, Clock)?',
        aliases=('side events', 'side event', 'pyraminx', 'megaminx', 'skewb',
                 'square-1', 'sq1', 'clock', 'puzzle events'),
    ),
    Attribute(
        key='is_fmc_specialist',
        kind='bool',
        question='Are they primarily a Fewest Moves solver?',
        aliases=('fmc', 'fewest moves'),
    ),
    Attribute(
        key='is_oh_specialist',
        kind='bool',
        question='Are they primarily a one-handed solver?',
        aliases=('one handed', 'oh', '3x3 one-handed'),
    ),
    Attribute(
        key='competed_all_events',
        kind='bool',
        question='Have they competed in every current WCA event?',
        aliases=('all events', 'every event'),
    ),
    Attribute(
        key='events_competed',
        kind='numeric',
        question='Have they competed in at least {value} different events?',
        aliases=('number of events', 'how many events'),
        thresholds=(8, 12, 15, 17),
    ),
]

# ---------------------------------------------------------------------------
# Volume — how much they've competed
# ---------------------------------------------------------------------------

_VOLUME = [
    Attribute(
        key='comp_count',
        kind='numeric',
        question='Have they competed in at least {value} competitions?',
        aliases=('competitions', 'how many competitions', 'comp count'),
        thresholds=(20, 40, 75, 150),
    ),
    Attribute(
        key='countries_competed_in',
        kind='numeric',
        question='Have they competed in at least {value} different countries?',
        aliases=('countries competed in', 'travelled', 'international'),
        thresholds=(2, 3, 5, 10),
    ),
    Attribute(
        key='total_results',
        kind='numeric',
        question='Do they have at least {value} recorded results?',
        aliases=('total results', 'number of solves'),
        thresholds=(100, 400, 800, 2000),
    ),
]

# ---------------------------------------------------------------------------
# Milestones — the facts a fan would actually know
# ---------------------------------------------------------------------------

_MILESTONES = [
    Attribute(
        key='sub10_333_single',
        kind='bool',
        question='Have they solved 3x3 in under 10 seconds at a WCA competition?',
        aliases=('sub 10 single', 'sub-10', 'under 10 seconds'),
    ),
    Attribute(
        key='sub10_333_average',
        kind='bool',
        question='Have they achieved a 3x3 average under 10 seconds at a WCA competition?',
        aliases=('sub 10 average', 'sub-10 average'),
    ),
    Attribute(
        key='sub6_333_single',
        kind='bool',
        question='Have they solved 3x3 in under 6 seconds at a WCA competition?',
        aliases=('sub 6', 'sub-6 single', 'under 6 seconds'),
        implies=('sub10_333_single',),
    ),
    Attribute(
        key='competed_at_worlds',
        kind='bool',
        question='Have they competed at a World Championship?',
        aliases=('worlds', 'world championship'),
    ),
    Attribute(
        key='has_worlds_podium',
        kind='bool',
        question='Have they finished in the top three in an event at a World Championship?',
        aliases=('worlds podium', 'world championship podium', 'world champion'),
        implies=('competed_at_worlds',),
    ),
]


ATTRIBUTES: tuple[Attribute, ...] = tuple(
    _IDENTITY + _ERA + _RECORDS + _RANKINGS + _EVENT_PROFILE + _VOLUME + _MILESTONES
)

BY_KEY: dict[str, Attribute] = {a.key: a for a in ATTRIBUTES}


# ---------------------------------------------------------------------------
# Predicates — the binary questions the engine actually asks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predicate:
    """A binary question: does this cuber satisfy `key <op> value`?

    This is also the shape the question parser emits, so a typed question and a
    generated question are the same object and flow through the same code.
    """

    key: str
    # 'is' (bool) | 'eq' (categorical) | 'gte' (numeric)
    # 'has' (multi contains one value) | 'has_group' (multi overlaps an event group)
    op: str
    value: object

    def test(self, attrs: dict) -> bool:
        """Evaluate against one row's `attrs` blob. Missing data reads as False."""
        actual = attrs.get(self.key)
        if actual is None:
            return False
        if self.op == 'is':
            return bool(actual) is bool(self.value)
        if self.op == 'eq':
            return actual == self.value
        if self.op == 'gte':
            try:
                return float(actual) >= float(self.value)
            except (TypeError, ValueError):
                return False
        if self.op == 'has':
            return self.value in (actual or ())
        if self.op == 'has_group':
            members = EVENT_GROUPS.get(str(self.value), ())
            return bool(set(actual or ()) & set(members))
        raise ValueError(f'unknown predicate op: {self.op}')

    def render(self) -> str:
        """The human-readable question text."""
        attr = BY_KEY.get(self.key)
        if attr is None:
            return f'{self.key} {self.op} {self.value}'
        return attr.question.format(value=_display(self.key, self.op, self.value))

    def as_id(self) -> str:
        """Stable identifier, used to remember which questions have been asked."""
        return f'{self.key}:{self.op}:{self.value}'


# WCA records gender as m / f / o, where 'o' is the profile's "Other" option.
# Rendering the raw code produced "Is the cuber o?" in live games.
_GENDER_WORDS = {'m': 'male', 'f': 'female', 'o': 'listed as another gender'}


def _display(key: str, op: str, value: object) -> str:
    """Prettify a raw stored value for display in a question."""
    if op == 'has_group':
        return GROUP_LABELS.get(str(value), str(value))
    if key == 'gender':
        return _GENDER_WORDS.get(str(value), str(value))
    if key == 'continent_id':
        # WCA continent ids carry a leading underscore: '_Europe'.
        return str(value).lstrip('_')
    if key == 'main_event_group':
        return GROUP_LABELS.get(str(value), str(value))
    if key in ('main_event', 'top100_events'):
        return event_label(str(value))
    if key == 'country_id':
        return country_label(str(value))
    return str(value)


def predicate_from_id(pred_id: str) -> Predicate:
    """Rebuild a Predicate from `as_id()`. Used to replay a stateless answer log."""
    key, op, raw = pred_id.split(':', 2)
    attr = BY_KEY.get(key)
    if attr is None:
        raise ValueError(f'unknown attribute: {key}')
    if op == 'is':
        value = raw == 'True'
    elif op == 'gte':
        value = int(raw)
    else:
        value = raw
    return Predicate(key, op, value)


def candidate_predicates(rows: list[dict]) -> list[Predicate]:
    """Every binary question worth asking about this candidate set.

    Categorical and multi values are drawn from the candidates themselves rather
    than a declared list, so the engine never asks "are they from Mongolia?" when
    no remaining candidate is Mongolian — the answer carries no information and
    the question looks broken to the player.
    """
    preds: list[Predicate] = []
    for attr in ATTRIBUTES:
        if attr.kind == 'bool':
            preds.append(Predicate(attr.key, 'is', True))
        elif attr.kind == 'numeric':
            preds.extend(Predicate(attr.key, 'gte', t) for t in attr.thresholds)
        elif attr.kind == 'categorical':
            seen = {r['attrs'].get(attr.key) for r in rows}
            seen.discard(None)
            preds.extend(Predicate(attr.key, 'eq', v) for v in sorted(seen, key=str))
        elif attr.kind == 'multi':
            seen: set[str] = set()
            for r in rows:
                seen.update(r['attrs'].get(attr.key) or ())
            preds.extend(Predicate(attr.key, 'has', v) for v in sorted(seen))
            if attr.event_valued:
                # Group questions narrow far faster than single-event ones when
                # the candidate set is still wide: "any big cube event" splits
                # four ways at once.
                preds.extend(
                    Predicate(attr.key, 'has_group', g)
                    for g, members in EVENT_GROUPS.items()
                    if seen & set(members)
                )
    return preds


# ---------------------------------------------------------------------------
# Relations — what one answer says about other attributes
# ---------------------------------------------------------------------------
#
# Attributes overlap heavily. "Their strongest event is 3x3" means they are not
# a one-handed specialist; "not world top 100 in anything" means not top 100 in
# 2x2 either; "they represent the USA" answers "are they from North America?".
# Before these relations were declared, a recorded game asked 39 questions and
# 17 of them repeated something the player had already said.
#
# A relation is written as clauses. A clause is a tuple of (predicate, truth)
# literals, and at least one literal in it holds for every cuber. "A implies B"
# becomes the clause (A is false, or B is true), which also covers the reverse
# case: if B is false, A is false. services/game/knowledge.py handles thresholds
# and categorical exclusivity itself, so a clause only has to name one threshold
# or one value on each side.
#
# Almost every relation follows from how scripts/build_cuber_profiles.py derives
# the attributes. The ones that depend on how the sport works instead are marked.
# A DB-backed test checks every clause against the full live table, so a builder
# change that breaks one fails there, not in a game.

Clause = tuple[tuple[Predicate, bool], ...]


def _is(key: str) -> Predicate:
    return Predicate(key, 'is', True)


def _eq(key: str, value: str) -> Predicate:
    return Predicate(key, 'eq', value)


def _gte(key: str, value: int) -> Predicate:
    return Predicate(key, 'gte', value)


def _top100(event: str) -> Predicate:
    return Predicate('top100_events', 'has', event)


def _top100_group(group: str) -> Predicate:
    return Predicate('top100_events', 'has_group', group)


def _implies(a: Predicate, b: Predicate, truth: bool = True) -> Clause:
    """If `a` holds, `b` has `truth`. Also: if `b` does not, `a` is false."""
    return ((a, False), (b, truth))


def _equivalent(a: Predicate, b: Predicate) -> list[Clause]:
    return [_implies(a, b), _implies(b, a)]


def _any_of(whole: Predicate, parts: list[Predicate]) -> list[Clause]:
    """`whole` holds exactly when at least one of `parts` does.

    With all of the parts ruled out, `whole` is false. With `whole` true and all
    but one part ruled out, the remaining part is true.
    """
    return [((whole, False), *((p, True) for p in parts)),
            *(_implies(p, whole) for p in parts)]


def _never_exceeds(lesser: str, greater: str) -> list[Clause]:
    """Numeric `lesser` is never above `greater`, so a bound on one bounds the other."""
    values = sorted(set(BY_KEY[lesser].thresholds) | set(BY_KEY[greater].thresholds))
    return [_implies(_gte(lesser, v), _gte(greater, v)) for v in values]


def _schema_relations() -> tuple[Clause, ...]:
    clauses: list[Clause] = []

    for attr in ATTRIBUTES:
        for target in attr.implies:
            clauses.append(_implies(_is(attr.key), _is(target)))

    # Records. Each "or better" flag is exactly the tier itself plus the ones
    # above it: has_cr_or_better is (wr + cr) > 0. So "NR or better, but no CR
    # or better" settles "at least one national record".
    clauses += _any_of(_is('has_wr'), [_gte('wr_single_count', 1),
                                       _gte('wr_average_count', 1)])
    clauses += _any_of(_is('has_cr_or_better'), [_is('has_wr'),
                                                 _gte('cr_count', 1)])
    clauses += _any_of(_is('has_nr_or_better'), [_is('has_cr_or_better'),
                                                 _gte('nr_count', 1)])
    # Rank 1 today came from a result that was flagged as a record when it was
    # set. Results are never deleted, so nobody had a better one at that point.
    # A change of nationality could break this in principle; it has never
    # happened in the data.
    clauses.append(_implies(_is('currently_wr'), _is('has_wr')))
    clauses.append(_implies(_is('currently_cr_or_better'), _is('has_cr_or_better')))
    clauses.append(_implies(_is('currently_nr_or_better'), _is('has_nr_or_better')))

    # Rankings. top100_event_count is the size of top100_events, and a top-10
    # rank is also a top-100 rank.
    any_top100 = _gte('top100_event_count', 1)
    two_top100 = _gte('top100_event_count', 2)
    events = sorted(EVENT_NAMES)
    for event in events:
        clauses.append(_implies(_top100(event), any_top100))
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            clauses.append(((_top100(a), False), (_top100(b), False), (two_top100, True)))
    for group, members in EVENT_GROUPS.items():
        clauses += _any_of(_top100_group(group), [_top100(e) for e in members])
        clauses.append(_implies(_top100_group(group), any_top100))
    clauses.append(_implies(_is('top10_any'), any_top100))

    # Strongest event. main_event is the event with their best world rank,
    # main_event_group is that event's group, and the specialist flags repeat
    # one or the other in different wording.
    for group, members in EVENT_GROUPS.items():
        clauses += _any_of(_eq('main_event_group', group),
                           [_eq('main_event', e) for e in members])
    # 2x2, 3x3, OH and FMC belong to no group, so they rule out every group.
    for event in events:
        if group_of(event) is None:
            clauses += [_implies(_eq('main_event', event),
                                 _eq('main_event_group', group), False)
                        for group in EVENT_GROUPS]
    for flag, same in (
        ('is_bld_specialist', _eq('main_event_group', 'blind_events')),
        ('is_bigcube_specialist', _eq('main_event_group', 'big_cubes')),
        ('is_sideevent_specialist', _eq('main_event_group', 'side_events')),
        ('is_fmc_specialist', _eq('main_event', '333fm')),
        ('is_oh_specialist', _eq('main_event', '333oh')),
    ):
        clauses += _equivalent(_is(flag), same)
    # If any of their ranks is in the world top 100, the best one is too.
    for event in events:
        clauses.append(((_eq('main_event', event), False), (any_top100, False),
                        (_top100(event), True)))
    for group in EVENT_GROUPS:
        clauses.append(((_eq('main_event_group', group), False), (any_top100, False),
                        (_top100_group(group), True)))

    # 3x3 milestones. The middle three solves of a sub-10 average include a
    # sub-10 single.
    clauses.append(_implies(_is('sub10_333_average'), _is('sub10_333_single')))
    # Depends on the sport: the current world top 100 in 3x3 is far faster than
    # either bar, single or average. These only get safer as times drop.
    clauses.append(_implies(_top100('333'), _is('sub6_333_single')))
    clauses.append(_implies(_top100('333'), _is('sub10_333_average')))

    # Career. A debut can't come after the latest competition, and every
    # competition adds at least one result.
    clauses += _never_exceeds('first_year', 'last_year')
    clauses += _never_exceeds('comp_count', 'total_results')
    # Competing in every one of the CURRENT_EVENTS means at least that many
    # distinct events. (Not the reverse: events_competed counts retired ones.)
    clauses.append(_implies(_is('competed_all_events'),
                            _gte('events_competed', len(CURRENT_EVENTS))))

    return tuple(clauses)


RELATIONS: tuple[Clause, ...] = _schema_relations()


def pool_relations(rows: list[dict]) -> list[Clause]:
    """Relations supplied by the data rather than the schema: country -> continent.

    The schema doesn't say which continent a country is in, but every row
    carries both values. A country that ever shows up under two continents gets
    no relation, so bad data can't produce a wrong one.
    """
    continents: dict[str, set] = {}
    for row in rows:
        attrs = row['attrs']
        country = attrs.get('country_id')
        if country is not None:
            continents.setdefault(country, set()).add(attrs.get('continent_id'))
    return [
        _implies(_eq('country_id', country), _eq('continent_id', next(iter(found))))
        for country, found in sorted(continents.items())
        if len(found) == 1 and None not in found
    ]
