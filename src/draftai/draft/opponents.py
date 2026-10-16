"""Opponent draft model: a conditional logit over the available players.

    P(team j picks p | S_t) = exp(u_jp) / sum_q exp(u_jq)
    u_jp = -beta(t) * (rank_t(p) - 1) + gamma * need_j(p)
    beta(t) = exp(c0) * t ** c1

with a small share ``eps`` of off-board picks mixed in (uniform over the available
players): keeper stashes, name-value reaches, autopicks after a timeout. Without it a
handful of such picks drag beta toward zero; with it the held-out likelihood on a real
Sleeper draft improves by about 0.7 nats per pick.

rank_t(p) is p's market (ADP) rank among the players still available, need_j(p) is 1 when
p would fill a starting slot team j cannot fill yet, and beta(t) is how closely drafters
follow the market at overall pick t (tight early, looser late when c1 < 0). The design's
news term is left out: historical drafts carry no dated news.

Parameters are fitted by maximum likelihood on real draft sequences (``fit``) or, with no
drafts, calibrated so simulated drafts reproduce the survival model's ADP spread
(``calibrate``). Survival odds in the rollouts are emergent outputs of this model, so its
calibration is the calibration of the "now or later" decision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from draftai.draft.needs import NeedModel


@dataclass
class OpponentModel:
    c0: float = -0.6
    c1: float = -0.45
    gamma: float = 0.8
    window: int = 50  # only the first `window` available players by market rank are considered
    eps: float = 0.0  # share of off-board picks
    reach: int = 150  # off-board picks are simulated as uniform over this many available players
    source: str = "default"
    diagnostics: dict = field(default_factory=dict)

    def beta(self, t: int) -> float:
        return float(np.exp(self.c0) * max(t, 1) ** self.c1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "OpponentModel":
        return cls(**json.loads(path.read_text()))

    def describe(self) -> str:
        b = ", ".join(f"beta({t})={self.beta(t):.2f}" for t in (1, 12, 50, 120))
        return f"opponents [{self.source}]: {b}; need gamma={self.gamma:.2f}; off-board picks {self.eps:.1%}"


class Market:
    """Board players in market (ADP) order, with their eligibility type.

    Every opponent array is kept in this column order, so "the first k available players
    by ADP" is a prefix of the columns.
    """

    def __init__(self, adp: np.ndarray, needs: NeedModel):
        self.adp = np.asarray(adp, dtype=float)
        self.order = np.argsort(self.adp, kind="stable")  # column -> board row
        self.col_of = np.empty_like(self.order)
        self.col_of[self.order] = np.arange(self.order.size)
        self.needs = needs
        self.type_col = needs.type_of[self.order]

    @property
    def size(self) -> int:
        return int(self.order.size)


def choose_batch(model: OpponentModel, market: Market, avail: np.ndarray, fills: np.ndarray, t: int,
                 u: np.ndarray) -> np.ndarray:
    """One opponent pick in each of n drafts at overall pick ``t``.

    ``avail`` is (n, columns) in market order, ``fills`` is (n, types) for the team on the
    clock, ``u`` is one uniform per draft (inverse-CDF sampling, so a shared ``u`` couples
    drafts that differ elsewhere: common random numbers). Returns columns, -1 if none.
    """
    if model.eps > 0:
        off = u < model.eps
        u = np.where(off, u / model.eps, (u - model.eps) / (1.0 - model.eps))
        if off.any():
            out = _choose_logit(model, market, avail, fills, t, u)
            out[off] = _choose_uniform(model, avail[off], u[off], t)
            return out
    return _choose_logit(model, market, avail, fills, t, u)


def _choose_uniform(model: OpponentModel, avail: np.ndarray, u: np.ndarray, t: int) -> np.ndarray:
    width = min(avail.shape[1], max(t - 1, 0) + model.reach)
    a = avail[:, :width]
    rank = np.cumsum(a, axis=1)
    n = np.minimum(rank[:, -1], model.reach)
    target = np.floor(u * n).astype(np.int64) + 1
    hit = a & (rank == target[:, None])
    return np.where(n > 0, hit.argmax(axis=1), -1)


def _choose_logit(model: OpponentModel, market: Market, avail: np.ndarray, fills: np.ndarray, t: int,
                  u: np.ndarray) -> np.ndarray:
    width = min(market.size, max(t - 1, 0) + model.window)
    a = avail[:, :width]
    rank = np.cumsum(a, axis=1)
    ok = a & (rank <= model.window)
    util = -model.beta(t) * (rank - 1.0) + model.gamma * fills[:, market.type_col[:width]]
    util = np.where(ok, util, -np.inf)
    top = util.max(axis=1, keepdims=True)
    has = np.isfinite(top[:, 0])
    p = np.exp(util - np.where(np.isfinite(top), top, 0.0))
    c = np.cumsum(p, axis=1)
    idx = (c < u[:, None] * c[:, -1:]).sum(axis=1)
    return np.where(has, np.minimum(idx, width - 1), -1)


def choose(model: OpponentModel, market: Market, avail: np.ndarray, counts: np.ndarray, t: int, u: float) -> int:
    """Single-draft version: ``avail`` (columns,), ``counts`` the picking team's type counts."""
    fills = market.needs.fills(counts)[None, :]
    return int(choose_batch(model, market, avail[None, :], fills, t, np.array([u]))[0])


def probabilities(model: OpponentModel, market: Market, avail: np.ndarray, counts: np.ndarray,
                  t: int) -> tuple[np.ndarray, np.ndarray]:
    """(columns, probabilities) for the team on the clock: what the mock draft shows."""
    cols = np.flatnonzero(avail)[: max(model.window, model.reach if model.eps else 0)]
    rank = np.arange(cols.size, dtype=float)
    u = -model.beta(t) * rank + model.gamma * market.needs.fills(counts)[market.type_col[cols]]
    u = np.where(rank < model.window, u, -np.inf)
    p = (1 - model.eps) * np.exp(u - logsumexp(u))
    if model.eps:
        p += model.eps / min(cols.size, model.reach) * (rank < model.reach)
    return cols, p


# --- fitting on real drafts -----------------------------------------------------------
@dataclass
class DraftRecord:
    """A real draft mapped onto board rows. ``row`` is None for players not on the board."""

    picks: list[tuple[int, int, int | None]]  # (pick_no, slot, board row)
    exclude_slots: set[int] = field(default_factory=set)  # e.g. your own seat
    label: str = ""


@dataclass
class _Example:
    t: int
    rank: np.ndarray
    need: np.ndarray
    chosen: int
    slot: object
    adp: float


def examples(record: DraftRecord, market: Market) -> list[_Example]:
    avail = np.ones(market.size, dtype=bool)
    counts: dict[int, np.ndarray] = {}
    out = []
    for pick_no, slot, row in sorted(record.picks):
        if row is None:
            continue
        col = int(market.col_of[row])
        c = counts.setdefault(slot, np.zeros(market.needs.n_types, dtype=np.int64))
        if slot not in record.exclude_slots and avail[col]:
            cols = np.flatnonzero(avail)
            k = int(np.searchsorted(cols, col))
            need = market.needs.fills(c)[market.type_col[cols]].astype(float)
            out.append(_Example(pick_no, np.arange(cols.size, dtype=float), need, k, slot, float(market.adp[row])))
        avail[col] = False
        c[market.type_col[col]] += 1
    return out


@dataclass
class _Batch:
    """Examples padded into matrices so the likelihood is one vectorised pass."""

    t: np.ndarray  # (n,)
    rank: np.ndarray  # (n, width), 0-based rank among the available players
    need: np.ndarray  # (n, width)
    mask: np.ndarray  # (n, width) real (not padding)
    chosen: np.ndarray  # (n,)
    size: np.ndarray  # (n,) players available

    @classmethod
    def of(cls, ex: list[_Example]) -> "_Batch":
        width = max(e.rank.size for e in ex)
        n = len(ex)
        rank, need, mask = np.zeros((n, width)), np.zeros((n, width)), np.zeros((n, width), dtype=bool)
        for i, e in enumerate(ex):
            k = e.rank.size
            rank[i, :k], need[i, :k], mask[i, :k] = e.rank, e.need, True
        return cls(np.array([e.t for e in ex], dtype=float), rank, need, mask,
                   np.array([e.chosen for e in ex]), mask.sum(axis=1))


def _nll(theta: np.ndarray, b: _Batch) -> float:
    """theta = (c0, c1, gamma, logit eps)."""
    c0, c1, gamma, le = theta
    eps = 1.0 / (1.0 + np.exp(-le))
    u = -np.exp(c0) * (b.t ** c1)[:, None] * b.rank + gamma * b.need
    u = np.where(b.mask, u, -np.inf)
    lp = u[np.arange(b.chosen.size), b.chosen] - logsumexp(u, axis=1)
    return float(-np.logaddexp(np.log1p(-eps) + lp, np.log(eps) - np.log(b.size)).sum())


NO_EPS = -30.0  # logit of a negligible off-board share
HELDOUT_FOLDS = 5


def _fit_theta(b: _Batch, free: tuple[bool, ...], start=(-0.6, -0.45, 0.8, -2.5)) -> np.ndarray:
    x0 = np.array(start, dtype=float)
    mask = np.array(free)

    def f(z):
        th = x0.copy()
        th[mask] = z
        return _nll(th, b)

    res = minimize(f, x0[mask], method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-3})
    th = x0.copy()
    th[mask] = res.x
    return th


def fit(records: list[DraftRecord], market: Market | list[Market], window: int = 50,
        heldout: bool = True) -> OpponentModel:
    """Maximum likelihood on every non-excluded pick, plus held-out comparisons.

    ``market`` is one market for every draft, or one per draft (drafts from different
    seasons each have their own market).

    Diagnostics report log-likelihood per pick for the full model, the model without the
    need term, a constant beta, no off-board picks and a uniform pick, each scored on
    held-out teams (fit on the other teams' picks, five folds) so extra parameters are not
    rewarded for fitting noise.
    """
    markets = market if isinstance(market, list) else [market] * len(records)
    ex = []
    for i, (r, m) in enumerate(zip(records, markets)):
        for e in examples(r, m):
            e.slot = (i, e.slot)  # held-out unit: one team in one draft
            ex.append(e)
    if len(ex) < 20:
        raise ValueError(f"only {len(ex)} usable picks; need at least 20 to fit opponents")
    # With flex and UTIL slots almost any player fills an open slot until a lineup is full,
    # after which none does: the need term is only identified by picks where it varies.
    need_varies = sum(1 for e in ex if 0 < e.need.mean() < 1)
    fit_gamma = need_varies >= 10
    variants = {"full": (True, True, fit_gamma, True), "no_need": (True, True, False, True),
                "constant_beta": (True, False, fit_gamma, True), "no_off_board": (True, True, fit_gamma, False)}
    starts = {"full": (-0.6, -0.45, 0.8, -2.5), "no_need": (-0.6, -0.45, 0.0, -2.5),
              "constant_beta": (-1.5, 0.0, 0.8, -2.5), "no_off_board": (-0.6, -0.45, 0.8, NO_EPS)}
    th = _fit_theta(_Batch.of(ex), variants["full"], starts["full"])
    teams = sorted({e.slot for e in ex}, key=str)
    folds = [teams[i::HELDOUT_FOLDS] for i in range(min(HELDOUT_FOLDS, len(teams)))]
    held: dict[str, float] = {k: 0.0 for k in variants}
    held["uniform"] = 0.0
    for fold in folds if heldout else []:
        train = _Batch.of([e for e in ex if e.slot not in fold])
        test_ex = [e for e in ex if e.slot in fold]
        test = _Batch.of(test_ex)
        for name, free in variants.items():
            held[name] += _nll(_fit_theta(train, free, starts[name]), test)
        held["uniform"] += sum(np.log(e.rank.size) for e in test_ex)
    n = len(ex)
    dev = np.array([e.t - e.adp for e in ex])
    diag = {
        "picks": n,
        "drafts": [r.label for r in records],
        "nll_per_pick_in_sample": _nll(th, _Batch.of(ex)) / n,
        "heldout_loglik_per_pick": {k: -v / n for k, v in held.items()} if heldout else None,
        "mean_abs_pick_minus_adp": float(np.abs(dev).mean()),
        "median_rank_of_pick_among_available": float(np.median([e.chosen + 1 for e in ex])),
        "picks_where_need_varies": need_varies,
        "gamma": "fitted" if fit_gamma else "not identified by these drafts; prior kept",
    }
    eps = float(1.0 / (1.0 + np.exp(-th[3])))
    return OpponentModel(float(th[0]), float(th[1]), float(th[2]), window, eps, source="fitted", diagnostics=diag)


# --- calibration without drafts -----------------------------------------------------------
def simulate_drafts(model: OpponentModel, market: Market, owners: list[int], n: int, seed: int) -> np.ndarray:
    """(n, picks) columns taken in n drafts where every team follows the model."""
    rng = np.random.default_rng(seed)
    teams = max(owners)
    avail = np.ones((n, market.size), dtype=bool)
    counts = np.zeros((n, teams + 1, market.needs.n_types), dtype=np.int64)
    out = np.full((n, len(owners)), -1, dtype=np.int64)
    rows = np.arange(n)
    for i, slot in enumerate(owners):
        fills = market.needs.fills_batch(counts[:, slot])
        col = choose_batch(model, market, avail, fills, i + 1, rng.random(n))
        ok = col >= 0
        avail[rows[ok], col[ok]] = False
        counts[rows[ok], slot, market.type_col[col[ok]]] += 1
        out[:, i] = col
    return out


def spread_by_band(picks: np.ndarray, market: Market, early: int) -> tuple[float, float]:
    """Mean |pick - ADP| over drafted players, for picks up to ``early`` and after."""
    t = np.broadcast_to(np.arange(1, picks.shape[1] + 1), picks.shape)
    ok = picks >= 0
    dev = np.abs(t[ok] - market.adp[market.order[picks[ok]]])
    e = t[ok] <= early
    return float(dev[e].mean()), float(dev[~e].mean())


def calibrate(market: Market, owners: list[int], sigma_a: float, sigma_b: float, gamma: float = 0.8,
              window: int = 50, n: int = 150, seed: int = 0, eps: float = 0.0) -> OpponentModel:
    """Choose beta(t) so simulated drafts match the configured ADP spread.

    Target: E|pick - ADP| = sigma(ADP) * sqrt(2 / pi), sigma = a + b * ADP, separately for
    the first three rounds and the rest.
    """
    teams = max(owners)
    early = 3 * teams
    picks_n = len(owners)
    adp_drafted = market.adp[market.order[:picks_n]]
    mad = (sigma_a + sigma_b * adp_drafted) * np.sqrt(2 / np.pi)
    t = np.arange(1, picks_n + 1)
    target = (float(mad[t <= early].mean()), float(mad[t > early].mean()))

    def loss(z):
        if not (-8.0 <= z[0] <= 4.0 and -2.5 <= z[1] <= 1.0):
            return 1e6
        m = OpponentModel(float(z[0]), float(z[1]), gamma, window, eps)
        got = spread_by_band(simulate_drafts(m, market, owners, n, seed), market, early)
        return sum((g / tg - 1.0) ** 2 for g, tg in zip(got, target))

    res = minimize(loss, np.array([-0.6, -0.45]), method="Nelder-Mead",
                   options={"xatol": 0.02, "fatol": 1e-4, "maxiter": 60})
    m = OpponentModel(float(res.x[0]), float(res.x[1]), gamma, window, eps, source="calibrated to ADP spread")
    got = spread_by_band(simulate_drafts(m, market, owners, n, seed + 1), market, early)
    m.diagnostics = {"target_mad_early_late": list(target), "simulated_mad_early_late": list(got),
                     "sigma_a": sigma_a, "sigma_b": sigma_b}
    return m
