# Directives for revising Section 3 (Results) — target: Journal of Computational Physics (JCP)

SCOPE: revise ONLY the "3. Results" section (3.1-3.5, all tables, all figure references/captions). Do not touch other sections. Preserve every number, every equation, every figure embed (`![](media/imageN.png)`), and every table with its numeric content exactly as given.

## JCP house style (paraphrased from the JCP/Elsevier Guide for Authors, applied to Section 3)
- Formal, objective, third-person-oriented scientific register; minimize first-person plural as a sentence subject.
- Quantitative, precise, hedge-appropriate claims; avoid promotional or subjective adjectives.
- Consistent notation and terminology throughout; define every symbol, abbreviation, and method-specific term at first use; use one English variant (American or British) consistently, not a mixture.
- Every figure and every table must be cited in the running text, in the order in which they appear, and each must be discussed rather than merely inserted.
- Tables should be used sparingly: numeric content shown in a table must not be re-narrated in prose elsewhere in the section (per JCP guidance that table data should not duplicate results described elsewhere in the article) — narrate the interpretation, not the raw values already tabulated.
- Mathematical variables are set in italics; displayed equations are numbered consecutively in the order cited in the text; cross-references to other parts of the paper must cite the specific subsection number (e.g., "Section 3.2"), never a vague "as discussed above/in the text."
- Reproducibility (ability of others to reproduce the reported results) is treated by JCP as a core evaluation criterion; retain and, where natural, foreground the manuscript's existing statements that support recomputation from stored results.

## Author persona (Writer agent)
Write as a researcher submitting to Journal of Computational Physics (JCP). Standards:
- Correct, natural English grammar and word order; native-level academic register.
- Do NOT coin idiosyncratic terminology. Use terminology already standard in the JFNK / Krylov-preconditioning / lattice Boltzmann literature (Schur complement, Jacobian-free Newton-Krylov, admissibility, residual, GMRES, etc.).
- Avoid "We"/"we" as a sentence subject. Use "This study ...", "The present work ...", passive constructions ("The method is evaluated ..."), or recast with the method/result as subject ("The results show ...", "Section 3.1 summarizes ..."). A few unavoidable instances are acceptable but must be rare (target: at most 2 in the entire section).
- No colloquial phrasing.
- Do NOT overstate or inflate the contribution ("state-of-the-art", "superior", "novel breakthrough", "dramatically", "significantly outperforms" used loosely, "far exceeds", etc.). Report the measured ratio/number and let it stand; every comparative claim must be tied to a specific number, case, or table already present.
- Write for a first-time reader of this manuscript: define every quantity, acronym, and method label at first use in Section 3, even if defined earlier in Section 2 (a brief in-line reminder is appropriate, e.g. "the operator-work metric (the number of native operator evaluations)").
- No repeated explanations, definitions, or numeric results within Section 3.
- Manuscript-specific terms (MSA-LBM, moment-Schur preconditioner, admissibility gate, time-to-threshold, operator work / LBE-call) must carry a short parenthetical gloss the FIRST time they appear in Section 3.

## Required organization of Section 3 (strict order, no shuffling)
1. Benchmark suite and methodology overview: families, grids, reference tiers, convergence/stopping criterion, baseline methods and settings, comparison protocol (time-to-threshold, operator-work metric).
2. Convergence results: convergence histories, plateau/robustness gap, shape of the convergence curves.
3. Performance of the proposed method relative to the other methods: quantitative wall-time and operator-work speedup comparisons.
4. Accuracy of the results: grid-convergence and reference comparisons.
Do not interleave; no forward references to a later block while explaining an earlier one.

## Additional constraints
- Keep subsection numbers 3.1-3.5 and the method/baseline names unchanged (MSA-LBM; native LBE, Anderson acceleration, preconditioned LBM, inexact Newton-Krylov, dual-time multigrid).
- No bold inline subheadings inside subsections; flowing prose.
- No defensive/audit-style padding ("independent verifier can check", repeated reassurances).
- English only.

OUTPUT: overwrite `solver_LBM_steady_state/english_paper/section3_only.md` with the revised Section 3. Return only a brief summary of changes (not the section text).
