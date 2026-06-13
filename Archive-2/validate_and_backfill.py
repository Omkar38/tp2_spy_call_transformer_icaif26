#!/usr/bin/env python3
"""
validate_and_backfill.py
========================

Companion to download_spy_options.py. Two jobs:

1. Validate the spy_options_daily/ dataset: rows, distinct sessions, and distinct
   contracts per year, plus a totals line.
2. Backfill any sessions that landed in skipped_days.txt (transient gateway 504s
   during the bulk pull) by re-pulling ONLY those days and merging them into the
   relevant year partition -- no need to rebuild whole years.

Usage
-----
    python validate_and_backfill.py                                   # validate only
    export DATABENTO_API_KEY=db-...   # (or --api-key / API_key.txt)
    python validate_and_backfill.py --backfill                        # validate + backfill skipped days

Backfill is deliberately gentle (default 2 concurrent requests) so it behaves on a
loaded gateway. Re-run it until skipped_days.txt is gone.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import download_spy_options as dl


def validate(out_dir: Path) -> pd.DataFrame:
    parts = sorted(out_dir.glob("year=*/data.parquet"))
    if not parts:
        print(f"no partitions under {out_dir}")
        return pd.DataFrame()
    rows = []
    for part in parts:
        year = int(part.parent.name.split("=")[1])
        df = pd.read_parquet(part, columns=["quote_date", "contract_id"])
        rows.append({"year": year, "rows": len(df),
                     "sessions": df["quote_date"].nunique(),
                     "contracts": df["contract_id"].nunique()})
    rep = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    print(rep.to_string(index=False))
    print(f"TOTAL  rows={rep['rows'].sum():,}  sessions={rep['sessions'].sum():,}")
    skipped = read_skipped(out_dir)
    print(f"skipped_days.txt: {len(skipped)} session(s)"
          + (f" -> {', '.join(map(str, skipped))}" if skipped else ""))
    return rep


def read_skipped(out_dir: Path) -> list[date]:
    return _read_dates(out_dir / "skipped_days.txt")


def _read_dates(p: Path) -> list[date]:
    if not p.exists():
        return []
    return sorted({date.fromisoformat(s.strip()) for s in p.read_text().splitlines() if s.strip()})


def _dataset_condition(cfg: dl.Config, lo: date, hi: date) -> dict[date, str]:
    """Per-day OPRA.PILLAR availability ('available' | 'degraded' | 'pending' | 'missing')."""
    recs = dl.make_client(cfg).metadata.get_dataset_condition(
        dataset=cfg.dataset, start_date=lo.isoformat(), end_date=(hi + timedelta(days=1)).isoformat())
    out: dict[date, str] = {}
    for r in recs:
        d = r.get("date") if isinstance(r, dict) else getattr(r, "date", None)
        c = r.get("condition") if isinstance(r, dict) else getattr(r, "condition", None)
        if d is None:
            continue
        try:
            out[date.fromisoformat(str(d)[:10])] = str(c)
        except ValueError:
            pass
    return out


def backfill(cfg: dl.Config, max_workers: int) -> None:
    days = read_skipped(cfg.out_dir)
    if not days:
        print("skipped_days.txt empty -- nothing to backfill")
        return
    print(f"checking {len(days)} skipped session(s)")

    # Source-side gaps: Databento flags some sessions 'missing'/'pending' (no OPRA data
    # exists). These can never be fetched -- record once and stop retrying every run.
    cond = _dataset_condition(cfg, min(days), max(days))
    gaps = sorted(d for d in days if cond.get(d, "available") in ("missing", "pending"))
    if gaps:
        kg = cfg.out_dir / "known_gaps.txt"
        kg.write_text("".join(f"{d}\n" for d in sorted(set(_read_dates(kg)) | set(gaps))))
        print(f"{len(gaps)} day(s) are OPRA source gaps (condition missing/pending) -> known_gaps.txt, "
              f"will not retry: {', '.join(map(str, gaps))}")
    fetchable = [d for d in days if d not in set(gaps)]

    if fetchable:
        close, divs = dl.fetch_spy_close_dividends(cfg)
        by_year: dict[int, list[date]] = {}
        for d in fetchable:
            by_year.setdefault(d.year, []).append(d)
        for year, ydays in sorted(by_year.items()):
            ydays = sorted(ydays)
            part = cfg.out_dir / f"year={year}" / "data.parquet"
            if not part.exists():
                # Never fabricate a missing year from a few days -- that would block a full pull.
                print(f"  {year}: NO partition yet -- run the full download for {year} first; "
                      f"keeping {len(ydays)} day(s)")
                continue
            base = pd.read_parquet(part)
            have = set(pd.to_datetime(base["quote_date"]).dt.date)
            need = [d for d in ydays if d not in have]
            if len(need) < len(ydays):
                print(f"  {year}: {len(ydays) - len(need)} day(s) already present -- dropping from list")
            if not need:
                continue
            # Narrow start/end so assemble_year only pulls the month(s) spanning the missing days.
            ncfg = dataclasses.replace(cfg, start=min(need), end=max(need), max_workers=max_workers)
            add = dl.assemble_year(dl.make_client(ncfg), ncfg, year, need, close, divs)
            got = set(add["quote_date"].unique()) if not add.empty else set()
            if add.empty:
                print(f"  {year}: still failing for {len(need)} day(s) (transient -- retry in a calm window)")
                continue
            add = add.drop(columns=["year"], errors="ignore")
            merged = (pd.concat([base, add], ignore_index=True)
                        .drop_duplicates(["quote_date", "contract_id"], keep="last")
                        .sort_values(["quote_date", "contract_id"]))
            merged.to_parquet(part, engine="pyarrow", index=False)
            print(f"  {year}: backfilled {len([d for d in need if d in got])} day(s) -> {len(merged):,} rows")

    # Pending = fetchable days not yet present in their partition. Source gaps are excluded
    # (they live in known_gaps.txt now), so the file converges to empty once the gateway cooperates.
    still = sorted(set(fetchable) - _resolved(cfg.out_dir, fetchable))
    f = cfg.out_dir / "skipped_days.txt"
    if still:
        f.write_text("".join(f"{d}\n" for d in still))
        print(f"{len(still)} fetchable session(s) still pending -- left in skipped_days.txt (re-run in a calm window)")
    else:
        f.unlink(missing_ok=True)
        print("all fetchable skipped sessions resolved; skipped_days.txt removed"
              + (f"  ({len(gaps)} permanent source gap(s) in known_gaps.txt)" if gaps else ""))


def _resolved(out_dir: Path, candidates: list[date]) -> set[date]:
    """Days from `candidates` now present in their year partition (i.e. successfully backfilled)."""
    done: set[date] = set()
    by_year: dict[int, list[date]] = {}
    for d in candidates:
        by_year.setdefault(d.year, []).append(d)
    for year, ydays in by_year.items():
        part = out_dir / f"year={year}" / "data.parquet"
        if not part.exists():
            continue
        have = set(pd.to_datetime(pd.read_parquet(part, columns=["quote_date"])["quote_date"]).dt.date)
        done |= {d for d in ydays if d in have}
    return done


def audit(cfg: dl.Config) -> None:
    """Rebuild the missing-day list from ground truth rather than the fragile append-log:
    NYSE sessions (SPY calendar) that Databento can serve ('available'/'degraded'), minus
    what is actually present in the partitions. Rewrites skipped_days.txt (backfillable days
    in existing partitions) and known_gaps.txt (source gaps); flags entirely-absent years."""
    close, _ = dl.fetch_spy_close_dividends(cfg)
    cal = sorted(d for d in close["quote_date"] if cfg.start <= d <= cfg.end)
    cond = _dataset_condition(cfg, cfg.start, cfg.end)
    gaps = sorted(d for d in cal if cond.get(d, "available") in ("missing", "pending"))
    expected = [d for d in cal if d not in set(gaps)]

    present: set[date] = set()
    years_present: set[int] = set()
    for part in cfg.out_dir.glob("year=*/data.parquet"):
        years_present.add(int(part.parent.name.split("=")[1]))
        present |= set(pd.to_datetime(pd.read_parquet(part, columns=["quote_date"])["quote_date"]).dt.date)

    by_year: dict[int, list[date]] = {}
    for d in expected:
        by_year.setdefault(d.year, []).append(d)

    backfillable: list[date] = []
    absent: list[int] = []
    for year, edays in sorted(by_year.items()):
        miss = sorted(set(edays) - present)
        if year not in years_present:
            absent.append(year)
            print(f"  {year}: ENTIRE YEAR ABSENT -- {len(edays)} sessions, run the full download")
        else:
            print(f"  {year}: {len(edays) - len(miss)}/{len(edays)} present"
                  + (f"  ->  {len(miss)} MISSING" if miss else ""))
            backfillable += miss

    f = cfg.out_dir / "skipped_days.txt"
    if backfillable:
        f.write_text("".join(f"{d}\n" for d in sorted(backfillable)))
    else:
        f.unlink(missing_ok=True)
    if gaps:
        (cfg.out_dir / "known_gaps.txt").write_text("".join(f"{d}\n" for d in gaps))
    print(f"\naudit: {len(backfillable)} backfillable missing day(s) -> skipped_days.txt; "
          f"{len(gaps)} source gap(s) -> known_gaps.txt")
    if absent:
        print(f"absent years needing a full download first: {absent}")


def _cfg_from_args(a: argparse.Namespace, out: Path) -> dl.Config:
    key = a.api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not key:
        kf = Path(__file__).with_name("API_key.txt")
        key = kf.read_text().split()[-1].strip() if kf.exists() else ""
    if not key:
        raise SystemExit("set DATABENTO_API_KEY, pass --api-key, or provide API_key.txt")
    return dl.Config(api_key=key, out_dir=out)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate, audit, and backfill the SPY options dataset.")
    p.add_argument("--out", default="./spy_options_daily")
    p.add_argument("--api-key", default=None)
    p.add_argument("--audit", action="store_true",
                   help="rebuild skipped_days.txt/known_gaps.txt from calendar vs partitions (ground truth)")
    p.add_argument("--backfill", action="store_true", help="re-pull sessions listed in skipped_days.txt")
    p.add_argument("--max-workers", type=int, default=2, help="concurrency for backfill (default 2, gentle)")
    a = p.parse_args()

    out = Path(a.out).expanduser().resolve()
    print("=== coverage ===")
    validate(out)

    if a.audit:
        print("\n=== audit (ground-truth missing days) ===")
        audit(_cfg_from_args(a, out))

    if a.backfill:
        print("\n=== backfill ===")
        backfill(_cfg_from_args(a, out), a.max_workers)
        print("\n=== coverage after backfill ===")
        validate(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
