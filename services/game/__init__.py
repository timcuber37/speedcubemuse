"""Guess the Cuber — an Akinator-style minigame over the WCA dataset.

The design principle worth knowing before reading any of these modules: the
LLM is not the game engine. Question *selection* is an information-gain search
over a precomputed feature matrix (`cuber_profiles`), which costs nothing per
turn. A model is only involved when a human types a free-form question and it
has to be mapped onto one of the attributes we already computed.

Module map:
    attributes.py       the attribute schema — single source of truth
    profiles.py         app-side read path + process cache
    engine.py           Bayesian candidate filter + question picker
    question_parser.py  free text -> predicate (the only LLM call)
"""
