# Editorial directives — bring the full paper to top-tier journal standard

TARGET: a top-tier computational-physics / CFD journal (J. Comput. Phys. / J. Fluid Mech. level).
LANGUAGE: the ENTIRE paper must be in English. The uploaded draft mixes English (Abstract, §1, §2) with Korean (parts of §4 Results, §6 Conclusion, the "부록" appendix). Translate ALL Korean into polished, idiomatic scientific English. No Korean characters may remain anywhere in the final document.

STRUCTURE — fix the section numbering and fill the gaps. The draft currently jumps 2 → 4 → 6 and has duplicated appendices. Produce this clean structure:
- Title, author, Abstract, Keywords
- 1. Introduction
- 2. Numerical Method (2.1–2.5, as in the draft — keep, light polish only; it is already revised)
- 3. Benchmark Suite and Evaluation Protocol  ← MUST CREATE. The material already exists but is misplaced at the top of "4. Results" (Table 5 benchmark families, the stopping protocol, baseline list, comparison-matching, reference tiers, fairness). Move all of that setup/protocol material out of Results into a proper §3, organized as flowing prose (no bold inline subheadings). Use the supplied §3 reference notes to ensure completeness; do not omit the benchmark-family table, the stopping criterion (macro-L2 residual + plateau + admissibility), the tolerance values, the five baselines with settings, and the time-to-threshold comparison definition.
- 4. Results  ← only actual results here (convergence, robustness, wall-time, operator-work, accuracy, ablation, memory). Translate the Korean subsections to English. Keep all numbers, tables, and figures.
- 5. Discussion  ← MUST CREATE from the supplied §5 reference notes (mechanistic interpretation + limitations and scope + future work). Concise, no defensive "threats to validity / verification-question" tables.
- 6. Conclusion (translate/condense the existing one)
- Appendix (merge the duplicated "부록 A/B" and "Appendix A.1/A.2" into ONE coherent appendix: full 27-case result table, supplementary convergence/diagnostic figures, cost-and-memory model, mass/flux diagnostics)
- References (keep as is)

PRESERVE EXACTLY:
- All displayed equations ($$...$$) and inline math. Do not drop or alter equations.
- All figure embeds. Keep every `![](media/imageN.png)` line (there are 11). Keep each figure's caption; translate captions to English. Do not renumber images in a way that breaks the media links.
- All tables and their numeric content. Re-render any pandoc grid-format tables as clean Markdown PIPE tables (header row + `|---|` separator) so they convert cleanly; translate table headers/captions to English. Do not change any numeric values.
- All reported quantitative claims and numbers exactly as in the draft. Do NOT invent, "improve", or recompute any number. If the abstract and a later section state numbers that appear inconsistent, do not silently change them — keep them and (only if clearly contradictory) note the inconsistency in the writer's summary, not in the paper body.

STYLE (top-tier):
- NO bold inline subheadings inside subsections; flowing academic prose.
- NO defensive/audit padding: remove repeated reassurances ("independent verifier can check", repeated "no reference injection / single routine / not curve fitting"), and do not add "threats to validity" or "anticipated reviewer concern" tables. State each such point once, briefly, where it naturally belongs.
- Avoid bare numbered cross-references that read awkwardly; prefer "as described above/below" or name the concept. Section cross-references are acceptable but use sparingly and correctly with the NEW numbering.
- Tight, precise, confident scientific voice. Improve clarity and flow throughout; fix any awkward phrasing, redundancy, or grammatical issues.
- Keep the method name consistent: "AP-Schur" (admissibility-preserving Schur); the running subtitle "Moment-Schur Accelerated LBM" may stay as the short title.

OUTPUT: write the complete revised paper to the file english_paper/revised_full.md (overwrite). Return only a short summary of what you changed — NOT the paper text.
