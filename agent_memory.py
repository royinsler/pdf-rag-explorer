"""Long-term memory for the verification agent: which retries actually work.

The agent's state (agent_state.py) lives for one question. This is what it carries
BETWEEN questions: for each kind of finding the checks can raise, how often a retry
resolved it. When history says retrying a kind almost never works, the loop stops
spending attempts on it and reports `degraded` straight away.

Why this is the memory worth having here, and the only one built:
  * it changes behaviour (fewer wasted calls), so it can be measured, not just claimed;
  * it is safe to keep. What is stored is a handful of counters per finding kind. No
    question, answer, quote, document text, page or file name is ever written - a
    contract's content cannot leak out of this file because none of it goes in. The
    kinds come from a fixed list; anything else is ignored, so a bug cannot smuggle
    text into the table. (Test: the database is dumped and searched.)

What it records, per (scope, kind):
  tries    runs where this kind was found on the first attempt AND a retry happened
  fixes    of those, runs where the kind was gone from the answer that was returned
  hopeless runs where this kind, when first seen, was already judged not worth retrying

The failure mode of any policy learned from its own outcomes is that it starves its
own data: once a kind looks hopeless it is never retried, so it can never look
otherwise. Two guards:
  * nothing is concluded before MIN_OBSERVATIONS tries - until then it always retries;
  * every PROBE_EVERY-th hopeless run retries anyway, so the numbers keep moving.

Honest status: the thresholds below are starting guesses, not measured values, and
this has run only against a scripted model. The first real use is to read `stats()`
and see whether the picture is sensible before trusting the policy.
"""
import sqlite3
from typing import Dict, Iterable, Optional

# The only strings that can ever be written as a "kind". Mirrors collect_findings.
KNOWN_KINDS = ("fabricated_quote", "unverified_fact", "garbage_echo", "damage_provenance")

MIN_OBSERVATIONS = 10     # tries before a kind may be judged hopeless
MIN_FIX_RATE = 0.10       # below this, with enough tries, a retry is not worth paying for
PROBE_EVERY = 10          # every Nth skip retries anyway, to keep the statistics fresh


class RetryMemory:
    """Per-finding-kind retry outcomes, in a small SQLite file."""

    def __init__(self, path: str, scope: str = "default", min_observations: int = MIN_OBSERVATIONS,
                 min_fix_rate: float = MIN_FIX_RATE, probe_every: int = PROBE_EVERY):
        self.path, self.scope = path, scope
        self.min_observations, self.min_fix_rate, self.probe_every = min_observations, min_fix_rate, probe_every
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS kind_stats ("
            " scope TEXT NOT NULL, kind TEXT NOT NULL,"
            " tries INTEGER NOT NULL DEFAULT 0, fixes INTEGER NOT NULL DEFAULT 0,"
            " hopeless INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (scope, kind))")
        self._db.commit()

    # -- reading --------------------------------------------------------------------

    def stats(self) -> Dict[str, Dict[str, int]]:
        rows = self._db.execute(
            "SELECT kind, tries, fixes, hopeless FROM kind_stats WHERE scope = ? ORDER BY kind",
            (self.scope,)).fetchall()
        return {kind: {"tries": tries, "fixes": fixes, "hopeless": hopeless}
                for kind, tries, fixes, hopeless in rows}

    def judged_hopeless(self, kinds: Iterable[str]) -> bool:
        """True when every known kind present has enough tries and a fix rate under the floor.

        The raw judgement, before the probe. Empty or unknown kinds are never hopeless.
        """
        kinds = [k for k in dict.fromkeys(kinds) if k in KNOWN_KINDS]
        if not kinds:
            return False
        stats = self.stats()
        for kind in kinds:
            row = stats.get(kind)
            if row is None or row["tries"] < self.min_observations:
                return False
            if row["fixes"] / row["tries"] >= self.min_fix_rate:
                return False
        return True

    def worth_retrying(self, kinds: Iterable[str]) -> bool:
        """Should the loop spend another attempt on findings of these kinds?

        False only when judged_hopeless - every kind present has enough tries and a fix
        rate under the floor - and this is not a probe run. Pure: it reads, never writes
        (outcomes are recorded at finalize), so re-running a node on resume cannot count
        twice.
        """
        kinds = [k for k in dict.fromkeys(kinds) if k in KNOWN_KINDS]
        if not self.judged_hopeless(kinds):
            return True
        stats = self.stats()
        return any((stats[k]["hopeless"] + 1) % self.probe_every == 0 for k in kinds)

    def describe(self, kinds: Iterable[str]) -> str:
        """Why the loop stopped, in numbers, for the report. No document content."""
        stats = self.stats()
        parts = []
        for kind in dict.fromkeys(kinds):
            row = stats.get(kind)
            if row and row["tries"]:
                parts.append(f"{kind}: {row['fixes']} of {row['tries']} past retries resolved it")
        return "; ".join(parts)

    # -- writing --------------------------------------------------------------------

    def record(self, first_kinds: Iterable[str], final_kinds: Iterable[str],
               retried: bool, hopeless: bool = False) -> None:
        """Record one finished run.

        first_kinds: finding kinds on the first attempt. final_kinds: kinds left in the
        answer that was returned. retried: whether a retry actually happened (it can in
        a hopeless run, as a probe). hopeless: judged_hopeless(first_kinds) at decision
        time; counted whether or not the probe retried, so probes are spaced evenly.
        """
        first = [k for k in dict.fromkeys(first_kinds) if k in KNOWN_KINDS]
        final = set(final_kinds)
        for kind in first:
            self._db.execute("INSERT OR IGNORE INTO kind_stats (scope, kind) VALUES (?, ?)",
                             (self.scope, kind))
            if hopeless:
                self._db.execute("UPDATE kind_stats SET hopeless = hopeless + 1 WHERE scope = ? AND kind = ?",
                                 (self.scope, kind))
            if retried:
                self._db.execute(
                    "UPDATE kind_stats SET tries = tries + 1, fixes = fixes + ? WHERE scope = ? AND kind = ?",
                    (0 if kind in final else 1, self.scope, kind))
        self._db.commit()

    def forget(self, scope: Optional[str] = None) -> None:
        """Delete everything remembered for this scope (the memory's own purge)."""
        self._db.execute("DELETE FROM kind_stats WHERE scope = ?", (scope or self.scope,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()
