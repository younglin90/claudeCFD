# dev_scratch/

One-off case-specific debug/sweep scripts from the pre-round-loop era (roughly June 22 - July 14),
archived here (2026-08-03) to declutter the repo root. Moved with `git mv`, not deleted -- full
history preserved, and each file is exactly where `git log --follow` will find it.

Not part of the round-loop's own protected reproduction-script set (`scripts/yadv_r*.py`, per
`.claude/skills/yadv-round/SKILL.md`'s own rule that those are permanent assets). Several of these
reference cases that are now excluded from the registered suite (case15: round 34; case24/33/34:
round 35) or investigate defects already fully diagnosed and documented in
`.claude/rules/denner-pitfalls.md` (case07's BDF2 wake wiggle, the dhat/MWI dead-end knobs).

Not touched by this cleanup: `paper_jcp.md`, `report.md`, `index.html` (live deliverables) and
their generator/upload scripts (`gen_report.sh`, `mkpaper.sh`, `plot_report.py`, `mk_errfigs.py`,
`chkfigs.sh`, `run_metrics.py`, `study_metrics.py`, `upload_*.sh`), which remain in the repo root.
