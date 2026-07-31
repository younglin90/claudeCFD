# SECOND aggressive-compression pass — directives

The previous pass cut only ~4%, which is insufficient. This pass MUST materially shorten the prose. Mandatory outcome: the total prose word count of the body (Sections 1–4) is reduced by AT LEAST 15% relative to the current file, achieved purely by removing redundancy and wordiness — never by deleting numbers, equations, figures, or tables.

PRESERVE EXACTLY (untouchable):
- Every numeric value, ratio, residual, tolerance, and reported result.
- Every equation, every section/subsection heading and number.
- All 11 figure embeds `![](media/imageN.png)` and captions; all tables with all numbers.
- Structure: Abstract; 1 Introduction; 2 Numerical Methods (2.1–2.5); 3 Results (3.1–3.5); 4 Conclusion; Appendix; Acknowledgements; References.
- MSA-LBM naming; English only.

MANDATORY CONSOLIDATIONS (these are required, not optional):
1. The lid-driven-cavity residual-floor narrative currently appears in Section 3.3, Section 3.4.1, and the Conclusion. State the floor phenomenon and its numbers ONCE (in 3.3, as evidence that non-convergence is stagnation), then in 3.4.1 refer to it in a single clause and give ONLY the new depth/timing numbers; in the Conclusion compress to a single clause with no re-quoted floor numbers. Remove every repeated sentence.
2. Section 3.4.1 is currently ~6 paragraphs (general per-baseline ratios; three-stage convergence; cavity floor; cavity depth+timing; cavity mechanism+timing; Couette/T-junction exceptions; summary). Compress to AT MOST 4 tighter paragraphs: (a) headline per-baseline and strict-subset ratios with Table 3a; (b) the three-stage convergence/Newton-collapse evidence, shortened; (c) the cavity advantage in depth and time (numbers once); (d) the exceptions (Couette, T-junction) and the slow-mode-dominance summary, merged. Keep all distinct numbers.
3. Section 3.4.2 (operator work): compress to AT MOST 2 paragraphs — the metric and the per-baseline/strongest-competitor result with the cavity example — folding in the cross-metric agreement in one sentence.
4. The long single-sentence three-stage chain in 3.4.1 must be broken into at most three sentences and shortened.
5. Trim the Abstract and Introduction of any sentence that repeats another; the Abstract should be one tight paragraph of background+method+results+scope plus the existing second results paragraph only if non-redundant.

ALSO (diction, already mostly done — keep enforcing): no throat-clearing, no filler adverbs, no rhetorical questions, standard academic verbs, consistent tense/hyphenation, idiomatic English.

DO NOT add new claims, numbers, or padding. Do not add bold inline subheadings.

OUTPUT: overwrite solver_LBM_steady_state/english_paper/revised_full.md. Return a short summary including the approximate before/after body word counts.
