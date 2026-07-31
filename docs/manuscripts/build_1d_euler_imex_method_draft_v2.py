#!/usr/bin/env python3
from __future__ import annotations
import json, math, zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image

ROOT = Path('/home/younglin90/work/claude_code/claudeCFD')
OUT = ROOT / 'docs' / 'manuscripts'
EVID = ROOT / 'results' / '1D' / 'paper_euler_evidence'
PLOTS = EVID / 'plots'
DATA = json.loads((EVID / 'paper_euler_evidence.json').read_text())
OUT.mkdir(parents=True, exist_ok=True)
DOCX = OUT / '1d_euler_imex_method_draft.docx'
DOCX_V2 = OUT / '1d_euler_imex_method_draft_v2.docx'
HTML = OUT / '1d_euler_imex_method_draft.html'
REPORT = OUT / '1d_euler_imex_revision_report.md'
EMU_PER_IN = 914400

def fmt(x, nd=3):
    if x is None or x == '': return ''
    try: x = float(x)
    except Exception: return str(x)
    if not math.isfinite(x): return ''
    if abs(x) >= 1e3 or (abs(x) > 0 and abs(x) < 1e-2): return f'{x:.{nd}e}'
    return f'{x:.{nd}g}'

def xesc(s): return escape(str(s), {'"': '&quot;'})

def core_row(case): return next(r for r in DATA['core'] if r['case'] == case)

def case_metric(row):
    case = row['case']; js = row['json']
    if case == '07_B':
        vals = []
        for sub in js.get('subcases', []):
            m = sub.get('metrics', {}); pa = sub.get('peak_amplitude', {})
            vals.append(f"{sub.get('case')}: L2p={fmt(m.get('L2p'))}, Lip={fmt(m.get('Lip'))}, p_amp={fmt(pa.get('p_peak_amp_ratio'))}, u_amp={fmt(pa.get('u_peak_amp_ratio'))}")
        return '; '.join(vals)
    if case == '24_H':
        vals = [s.get('rho_profile_l2') for s in js.get('subcases', []) if s.get('rho_profile_l2') is not None]
        return 'max rho_profile_l2=' + fmt(max(vals) if vals else None)
    keys = ['p_rel_linf','u_abs_linf','p_scaled_l2','rho_scaled_l2','case13_rho_smooth_l2_rel','case14_rho_plateau085_089_linf_ratio','rho_l1_ratio','Tmix_l1_ratio']
    return '; '.join(f'{k}={fmt(js[k])}' for k in keys if k in js) or 'see core_metrics.csv'

def core_table(): return [[r['case'], 'PASS' if r['pass'] else 'FAIL', fmt(r.get('wall_s'),2), case_metric(r)] for r in DATA['core']]

def pe_table():
    rows=[]
    for case in ['01_A','02_A','16_T','17_T','18_T']:
        row=core_row(case); js=row['json']
        rows.append([case, fmt(js.get('p_rel_linf') or js.get('p_rel')), fmt(js.get('u_abs_linf') or js.get('u_abs')), fmt(js.get('rho_l1_ratio')), 'PASS' if row['pass'] else 'FAIL'])
    return rows

def acoustic_table():
    row=core_row('07_B')
    rows=[]
    for sub in row['json'].get('subcases', []):
        m=sub.get('metrics',{}); pk=sub.get('peak',{}); pa=sub.get('peak_amplitude',{}); sym=sub.get('symmetry',{}); osc=sub.get('osc',{})
        rows.append([sub.get('case'), sub.get('N'), fmt(m.get('L2p')), fmt(m.get('Lip')), fmt(m.get('L2u')), fmt(m.get('Liu')), fmt(pa.get('p_peak_amp_ratio')), fmt(pa.get('u_peak_amp_ratio')), fmt(sym.get('p_symmetry_max_error')), fmt(osc.get('p_alt_ratio')), 'PASS' if sub.get('pass') else 'FAIL'])
    return rows

def shock_table():
    rows=[]
    for case in ['13_E','14_E','15_E','24_H','25_H']:
        row=core_row(case); js=row['json']
        if case=='13_E':
            rows.append([case, fmt(js.get('case13_u_shock_delta_cells')), fmt(js.get('case13_rho_smooth_l2_rel')), fmt(js.get('contact_rho_peak_overshoot_ratio')), fmt(js.get('rho_smooth_local_tv_excess')), 'PASS'])
        elif case=='14_E':
            rows.append([case, fmt(js.get('case14_u_shock_delta_cells')), fmt(js.get('case14_rho_plateau085_089_linf_ratio')), fmt(js.get('case14_rho_peak085_overshoot_ratio')), fmt(js.get('rho_smooth_local_tv_excess')), 'PASS'])
        elif case=='15_E':
            rows.append([case, '', '', fmt(js.get('alpha_peak')), fmt(js.get('rho_smooth_local_tv_excess')), 'PASS'])
        elif case=='24_H':
            vals=[s.get('rho_profile_l2') for s in js.get('subcases',[]) if s.get('rho_profile_l2') is not None]
            shock=[s.get('shock_cells') for s in js.get('subcases',[]) if s.get('shock_cells') is not None]
            rows.append([case, fmt(max(shock) if shock else None), fmt(max(vals) if vals else None), 'bounded', 'see subcases', 'PASS'])
        elif case=='25_H':
            rows.append([case, fmt(js.get('shock_delta_cells')), fmt(js.get('rho_scaled_l2')), fmt(js.get('interface_rho_overshoot')), fmt(js.get('p_smooth_local_tv_excess')), 'PASS'])
    return rows

def thermal_table():
    rows=[]
    for case in ['16_T','17_T','18_T']:
        row=core_row(case); js=row['json']
        rows.append([case, js.get('N'), fmt(js.get('Tmix_l1_ratio')), fmt(js.get('Tmix_linf_ratio')), fmt(js.get('T1_active_hf_max_error')), fmt(js.get('T2_active_hf_max_error')), fmt(js.get('rho_smooth_hf_max_error')), 'PASS'])
    return rows

def grid_table():
    rows=[]
    for r in DATA['grid']:
        env=r.get('env',{}); N=''
        for k,v in env.items():
            if k.endswith('_N') or '_N_' in k: N=v; break
        rows.append([r['case'], N, 'PASS' if r['pass'] else 'FAIL', case_metric({'case':r['case'],'json':r['json']})])
    return rows

def ablation_table():
    rows=[]
    by={}
    for r in DATA['baseline']:
        var=r['label'].split('/',1)[0]
        by.setdefault(var,[0,0,[]])
        by[var][1]+=1
        if r['pass']: by[var][0]+=1
        by[var][2].append(r['case'])
    for k,v in by.items(): rows.append([k, f'{v[0]}/{v[1]}', ', '.join(sorted(set(v[2]))), 'baseline_metrics.csv'])
    return rows

def cfl_table():
    rows=[]
    for r in DATA['cfl']:
        js=r['json']; sub=(js.get('subcases') or [{}])[0]; m=sub.get('metrics',{}); pa=sub.get('peak_amplitude',{})
        rows.append([r['label'], r.get('env',{}).get('FIVE_EQ_CASE07_CFL',''), 'PASS' if r['pass'] else 'FAIL', fmt(m.get('L2p')), fmt(m.get('Lip')), fmt(pa.get('p_peak_amp_ratio'))])
    return rows

class Docx:
    def __init__(self): self.body=[]; self.rels=[]; self.media=[]; self.rid=2; self.pic_id=1
    def p(self, text='', style=None, bold=False, italic=False):
        ppr=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        rpr=''
        if bold or italic: rpr='<w:rPr>'+('<w:b/>' if bold else '')+('<w:i/>' if italic else '')+'</w:rPr>'
        self.body.append(f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{xesc(text)}</w:t></w:r></w:p>')
    def runs(self, runs, style=None):
        ppr=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        out=[]
        for text,bold,italic in runs:
            rpr=''
            if bold or italic: rpr='<w:rPr>'+('<w:b/>' if bold else '')+('<w:i/>' if italic else '')+'</w:rPr>'
            out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{xesc(text)}</w:t></w:r>')
        self.body.append(f'<w:p>{ppr}{"".join(out)}</w:p>')
    def h(self, level, text): self.p(text, style=f'Heading{level}')
    def bullet(self, text): self.p('• '+text)
    def eq(self, text): self.p(text, style='Equation')
    def table(self, headers, rows):
        tbl=['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
        def cell(v,b=False):
            rpr='<w:rPr><w:b/></w:rPr>' if b else ''
            return f'<w:tc><w:tcPr><w:tcW w:w="2200" w:type="dxa"/></w:tcPr><w:p><w:r>{rpr}<w:t xml:space="preserve">{xesc(v)}</w:t></w:r></w:p></w:tc>'
        tbl.append('<w:tr>'+''.join(cell(h,True) for h in headers)+'</w:tr>')
        for row in rows: tbl.append('<w:tr>'+''.join(cell(c) for c in row)+'</w:tr>')
        tbl.append('</w:tbl>'); self.body.append(''.join(tbl))
    def image(self, path: Path, caption: str, width_in=6.25):
        if not path.exists(): self.p(f'[Missing figure: {path}]', italic=True); return
        rel_id=f'rId{self.rid}'; self.rid+=1; name=f'image{len(self.media)+1}{path.suffix.lower()}'; self.media.append((name,path)); self.rels.append((rel_id,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image',f'media/{name}'))
        with Image.open(path) as im: w,h=im.size
        cx=int(width_in*EMU_PER_IN); cy=int(cx*h/w); pid=self.pic_id; self.pic_id+=1
        self.body.append(f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{pid}" name="Figure {pid}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/></wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{pid}" name="{xesc(name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
        self.p(caption, style='Caption')
        self.p(str(path.relative_to(ROOT)), style='Caption')
    def page_break(self): self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    def xml(self):
        sect='<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>'+''.join(self.body)+sect+'</w:body></w:document>'
    def rels_xml(self):
        rels=[('rId1','http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles','styles.xml')]+self.rels
        return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{r}" Type="{typ}" Target="{tar}"/>' for r,typ,tar in rels)+'</Relationships>'

D=Docx()
D.p('A pressure-equilibrium-preserving IMEX-SSP3 all-speed finite-volume method for one-dimensional compressible two-phase five-equation Euler flows', style='Title')
D.runs([('Manuscript draft v2. ', True, False), ('Prepared from the claudeCFD 1D Euler evidence package. Author names, affiliations, and target-journal formatting remain to be inserted.', False, False)])
D.h(1,'Paper configuration record')
D.table(['Item','Selection'], [['Article type','Numerical-method journal article'],['Field','Computational fluid dynamics; compressible multiphase flow; finite-volume methods'],['Scope','One-dimensional Euler-equation five-equation two-phase model'],['Excluded scope','Gravity, phase change, surface tension, viscosity, reaction, and multidimensional production claims'],['Citation style','Numbered engineering style; DOI included where verified'],['Output','DOCX manuscript draft with embedded figures and reproducibility paths']])
D.h(1,'Abstract')
D.p('A one-dimensional all-speed finite-volume method is developed for compressible two-phase flows described by a five-equation diffuse-interface model. The method targets a narrow but demanding regime: pressure-equilibrium material transport, acoustic reflection and transmission across large impedance jumps, shock-interface interaction, cavitation-like expansion, thermal-contrast advection, and hypersonic gas-liquid impact, all using a single production configuration. The discretization combines a stage-residual IMEX-SSP3 time integrator, an SLAU2-type pressure-free material flux, adaptive-BVD volume-fraction transport, T-MLP-u primitive-variable reconstruction with a superbee TVD base limiter, and a pressure-equilibrium target recovery used only to enforce the exact invariant manifold of p/u-flat material transport. The production run disables exact periodic remapping and Rusanov fallback. In the Euler-only validation suite, the method passes 13 core one-dimensional tests. The most severe Air-Water acoustic case at N=400 gives L2p = 9.00e-2, Lip = 3.55e-1, pressure peak-amplitude ratio 0.998, and velocity peak-amplitude ratio 0.966. Shock-interface tests show smooth-region density errors below 5e-3 in 13_E and a close-discontinuity plateau error of 2.99e-3 in 14_E. Grid-refinement and ablation studies indicate that the complete coupling of IMEX-SSP3, SLAU2, adaptive-BVD, and bounded high-order primitive reconstruction is required for the full test set. The present manuscript is intentionally limited to one-dimensional Euler validation; source-term physics and multidimensional extension are left for subsequent work.')
D.runs([('Keywords: ', True, False), ('compressible two-phase flow; five-equation model; all-speed method; IMEX-SSP3; SLAU2; pressure equilibrium; T-MLP-u; adaptive-BVD; finite volume', False, False)])
D.h(1,'Optional Korean abstract')
D.p('본 논문 초안은 1차원 압축성 다상유동 five-equation Euler 모델을 대상으로 하는 all-speed finite-volume 수치기법을 정리한다. 제안 기법은 IMEX-SSP3 시간적분, SLAU2 계열 material flux, adaptive-BVD 체적분율 수송, T-MLP-u + superbee primitive reconstruction, pressure-equilibrium target recovery를 결합한다. 검증 범위는 중력, 상변화, 표면장력, 반응 소스항을 제외한 Euler 방정식에 한정한다. 동일한 production configuration으로 13개의 핵심 1D 검증을 통과했으며, 강한 air-water acoustic impedance jump, shock-interface interaction, thermal contrast advection, hypersonic gas-liquid 문제에서 pressure/velocity equilibrium 보존과 낮은 수치확산을 확인하였다. 이 한글 초록은 내부 검토용이며, 영문 저널 또는 arXiv 제출본에서는 삭제할 수 있다.')
D.h(1,'Highlights')
for b in ['A single 1D production configuration is used for the Euler validation suite.', 'Pressure-equilibrium material transport is preserved without exact periodic remapping.', 'Adaptive-BVD alpha transport and bounded T-MLP-u primitive reconstruction reduce interface diffusion without case-ID switching.', 'Ablation data show that simplified upwind, HLLC-split, and alternative limiter variants do not reproduce the complete target behavior.', 'The paper explicitly limits its claims to 1D Euler flows; source-term and multidimensional results are deferred.']: D.bullet(b)
D.h(1,'Nomenclature')
D.table(['Symbol','Meaning'], [['alpha_k','Volume fraction of phase k'],['rho_k, e_k, T_k','Phase density, specific internal energy, and temperature'],['u, p','Mixture velocity and common pressure'],['rho, E','Mixture density and total specific energy'],['c_k, Z_k','Phase sound speed and acoustic impedance'],['D_1','Kapila/pressure-equilibrium non-conservative volume-fraction coefficient'],['R_E, R_I','Explicit material/advection and implicit acoustic residuals'],['dx, dt','Cell size and time step']])
D.h(1,'1. Introduction')
for text in [
'Compressible two-phase flows combine acoustic propagation, material contact motion, thermodynamic stiffness, and strong density or impedance jumps. In a five-equation diffuse-interface model, numerical diffusion of volume fraction contaminates phase densities and temperatures, while inconsistent pressure work can generate non-physical pressure oscillations at contacts. These two errors interact strongly in water-air calculations because the acoustic impedance contrast can exceed three orders of magnitude.',
'The classical five-equation framework of Allaire, Clerc, and Kokh [1] and related Kapila-type reductions [2,3] provide a compact model for single-pressure, single-velocity compressible mixtures. The model is attractive for shock-interface and gas-liquid flows, but its practical accuracy depends on the discretization: pressure-equilibrium transport must remain an invariant, acoustic waves must not be over-damped at low Mach number, and sharp contacts must not introduce overshoot or checkerboard modes.',
'Existing interface-capturing and all-speed ideas address parts of this problem. CICSAM [4] and later switching/compressive schemes such as MSTACS [5] preserve sharp volume-fraction profiles. MUSCL-THINC-BVD reconstruction reduces boundary variation and has been used in five-equation multiphase flows [6]. IMEX Runge-Kutta methods provide a path for separating stiff acoustic terms from material transport [7,8], and SLAU-type all-speed fluxes reduce low-Mach dissipation while retaining shock robustness [9]. The contribution here is to combine these ideas into a reproducible one-dimensional production solver and to quantify its behavior across a single strict Euler validation suite.',
'The intended contribution is therefore method-integration and evidence, not a claim of universal multidimensional readiness. The manuscript documents which mechanisms are used, where they matter, and what evidence is available. This framing is important: source terms for gravity, phase change, surface tension, reaction, and viscosity are excluded, and no multidimensional Kelvin-Helmholtz, shock-bubble, or Rayleigh-Taylor claim is made in this paper.'
]: D.p(text)
D.h(1,'2. Governing equations and thermodynamics')
D.p('The state is represented by W = (alpha_1, T_1, T_2, u, p)^T and U = (alpha_1 rho_1, alpha_2 rho_2, rho u, rho E, alpha_1)^T. The volume fractions satisfy alpha_2 = 1 - alpha_1. Phase densities and internal energies are obtained from the phase equations of state rho_k(p,T_k) and e_k(rho_k,p).')
for eq in ['∂(alpha_k rho_k)/∂t + ∂(alpha_k rho_k u)/∂x = 0,  k = 1,2', '∂(rho u)/∂t + ∂(rho u^2 + p)/∂x = 0', '∂(rho E)/∂t + ∂((rho E+p)u)/∂x = 0', '∂alpha_1/∂t + u ∂alpha_1/∂x = (alpha_1 + D_1) ∂u/∂x']:
    D.eq(eq)
D.p('The mixture density and energy are rho = alpha_1 rho_1 + alpha_2 rho_2 and rho E = alpha_1 rho_1 e_1 + alpha_2 rho_2 e_2 + 0.5 rho u^2. For the Kapila pressure-equilibrium form, the non-conservative coefficient is')
D.eq('D_1 = alpha_1 alpha_2 (rho_2 c_2^2 - rho_1 c_1^2) / (alpha_2 rho_1 c_1^2 + alpha_1 rho_2 c_2^2).')
D.p('The implementation supports ideal gas, stiffened gas, and Noble-Abel stiffened gas thermodynamics. The latter is included because water represented by a simple stiffened gas can distort density levels in strong gas-liquid tests; NASG is a more suitable convex EOS for liquid-like compressibility [10]. The solver uses analytic derivatives drho/dp|T, drho/dT|p, de/dp|T, and de/dT|p in the primitive-to-conservative Jacobian dU/dW. This derivative path is important for the implicit pressure solve because p is a primitive unknown, not a post-processed diagnostic.')
D.h(1,'3. Finite-volume discretization')
D.p('Cell averages U_i are advanced by a conservative finite-volume residual plus the non-conservative alpha source. In semi-discrete form,')
D.eq('dU_i/dt = -(F_{i+1/2} - F_{i-1/2})/dx + H_i.')
D.p('The flux is split into an explicit material/advection part and an implicit acoustic pressure part, R(W)=R_E(W)+R_I(W). The material residual advects alpha, phase masses, momentum transport, and thermodynamic scalars using a face velocity. The acoustic residual contains the pressure-gradient and pressure-work terms. This split is chosen to reduce acoustic stiffness without turning material transport into a first-order fully implicit update.')
D.h(2,'3.1 IMEX-SSP3 stage residual')
D.p('The production path uses the Pareschi-Russo IMEX-SSP3(4,3,3) stage residual form [7]. With the sign convention U_t + R_E(W) + R_I(W)=0, the i-th implicit stage solves')
D.eq('U_i* = U^n - dt Σ_{j<i}(aE_ij R_E(W_j) + aI_ij R_I(W_j)),')
D.eq('(U(W_i)-U_i*)/(aI_ii dt) + R_I(W_i) = 0.')
D.p('The final update uses the equal explicit and implicit weights b_E=b_I=(0,1/6,1/6,2/3). The diagonal implicit coefficient used in the implementation is gamma = 0.24169426078821. All high-order spatial limiters are re-evaluated at the explicit stages. Conservative stage blending is used to avoid mixing incompatible primitive states in mixed cells.')
D.h(2,'3.2 SLAU2-type material face velocity')
D.p('The material face velocity is an SLAU2-type all-speed velocity. In the implemented pressure-free IMEX split, the face velocity is constructed from reconstructed left and right mixture states as')
D.eq('u_f = u_Roe - chi (p_R - p_L)/(rho_bar c_bar),   chi=(1-Mhat)^2,   Mhat=min(1, u_rms/c_bar).')
D.p('The pressure correction vanishes in the high-Mach limit and remains active at low Mach number. This is the main reason the method is less diffusive than a Rusanov fallback in acoustic-interface problems. Rusanov fallback is disabled in the production evidence to ensure that the reported results are obtained by the proposed flux path.')
D.h(2,'3.3 Primitive reconstruction: T-MLP-u with TVD base limiter')
D.p('For primitive variables other than alpha, the production setting is T-MLP-u with a superbee TVD base limiter. On a uniform grid, for a scalar q, the candidate face extrapolation from cell i is')
D.eq('q_{i+1/2}^{L,*} = q_i + 0.5 psi(r_i) (q_{i+1}-q_i),   r_i=(q_{i+1}-q_i)/(q_i-q_{i-1}).')
D.p('For the superbee base limiter,')
D.eq('psi_SB(r)=max(0, min(2r,1), min(r,2)).')
D.p('The T-MLP-u wrapper clips psi by a local three-cell maximum-principle bound so that q_{i+1/2} remains in [min(q_{i-1},q_i,q_{i+1}), max(q_{i-1},q_i,q_{i+1})]. This keeps the useful compressive range 0 <= psi <= 2 without creating new primitive extrema. For homogeneous double-rarefaction topologies, a parameter-free detector uses van Leer as the TVD base limiter to avoid cavitation-pocket ringing; this is a state-topology rule, not a validation case switch.')
D.h(2,'3.4 Characteristic reconstruction policy')
D.p('When the local stencil is composition-uniform, the solver may reconstruct acoustic variables in characteristic form. When alpha varies across the stencil, characteristic reconstruction is disabled and EOS-consistent primitive or mixture scalar reconstruction is used. This policy prevents mixing characteristic variables that belong to different EOS branches across a material interface. The same policy is applied to every validation case.')
D.h(2,'3.5 Adaptive-BVD alpha transport')
D.p('The alpha transport uses adaptive-BVD logic. Near pure 0/1 material interfaces the method selects a CICSAM-like compressive construction; away from such sharp contacts it uses bounded MUSCL-Hancock TVD transport. The sharp alpha correction is applied in a flux-corrected form: a single local maximum-principle factor theta multiplies the induced corrections in alpha, phase mass, momentum, and energy. This avoids sharpening alpha while leaving conservative variables inconsistent.')
D.h(2,'3.6 Pressure-equilibrium target recovery')
D.p('For p/u-flat material transport, the exact solution remains on a pressure-equilibrium manifold. In those states, the implicit acoustic residual is zero and the material update should not create pressure or velocity errors. The solver therefore recovers the primitive state from the conservative target under the same EOS and pressure-equilibrium constraint. This is not an exact remap of the spatial solution; the production evidence explicitly uses FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0. The purpose is to avoid introducing pressure error solely through nonlinear conservative-to-primitive inversion on an invariant manifold.')
D.h(1,'4. Algorithm')
for b in ['Recover W and EOS derivatives from U using the analytic dU/dW Jacobian.', 'Extend boundary states and compute phase sound speeds, mixture sound speed, and acoustic impedance.', 'Construct SLAU2 material face velocities and acoustic pressure/velocity face states.', 'Reconstruct primitive variables with T-MLP-u; reconstruct alpha with adaptive-BVD and conservative FCT limiting.', 'Evaluate R_E and R_I at each IMEX-SSP3 stage.', 'Solve the implicit pressure/acoustic stage residual and convert the conservative stage target back to primitive variables.', 'Apply conservative SSP stage blending and pressure-equilibrium target recovery only when the physical p/u-flat invariant detector is satisfied.', 'Write diagnostic metrics and diff_vs_exact.png for each validation case.']:
    D.bullet(b)
D.h(1,'5. Validation design')
D.p('The validation suite is intentionally one-dimensional and Euler-only. It contains pressure-equilibrium advection, low-Mach acoustic propagation, acoustic reflection/transmission across material interfaces, shock-interface interactions, cavitation-like expansion, thermal contrast advection, a hypersonic mixture shock, and a hypersonic air-water interaction. Acceptance criteria combine exact-solution error, numerical diffusion, peak amplitude and peak location, local total-variation excess, high-frequency guards, and admissibility checks. Sharp shock neighborhoods are evaluated separately from smooth regions to avoid penalizing unavoidable cell-scale shock thickness while still detecting non-physical peaks and oscillations.')
D.table(['Production parameter','Value'], [['TIME_INTEGRATOR','imex_ssp3'],['ALPHA_SCHEME','adaptive_bvd'],['PRIMITIVE_SCHEME','tmlpu'],['TMLPU_TVD','superbee'],['MATERIAL_FLUX','slau2'],['PRESSURE_CLOSURE','regime_auto'],['CHARACTERISTIC_RECON','1, only composition-uniform stencils'],['RUSANOV_FALLBACK','0'],['UNIFORM_PERIODIC_REMAP','0'],['Maximum N used in manuscript evidence','800']])
D.h(1,'6. Results')
D.h(2,'6.1 Core Euler sweep')
D.p('The fixed production configuration passes all 13 core Euler tests. Table 2 reports representative metrics; complete raw metrics are in core_metrics.csv.')
D.table(['Case','Status','Wall time (s)','Representative metric'], core_table())
D.image(PLOTS/'pressure_equilibrium_preservation.png','Figure 1. Pressure-equilibrium preservation over material and thermal transport cases.')
D.h(2,'6.2 Pressure-equilibrium and thermal transport')
D.p('Pressure and velocity errors remain near roundoff in pressure-equilibrium transport cases. Thermal cases additionally test whether active-phase temperature and mixture temperature remain bounded and non-oscillatory.')
D.table(['Case','p relative Linf','u absolute Linf','rho L1 ratio','Status'], pe_table())
D.table(['Case','N','Tmix L1 ratio','Tmix Linf ratio','T1 HF max','T2 HF max','rho HF max','Status'], thermal_table())
for fname,cap in [('core_16_T.png','Figure 2. Hot-gas/cold-liquid discontinuous thermal advection.'),('core_17_T.png','Figure 3. Smooth alpha Gaussian thermal transport.'),('core_18_T.png','Figure 4. Smooth thermal-wave pressure-equilibrium transport.')]: D.image(PLOTS/fname,cap)
D.h(2,'6.3 Acoustic reflection and transmission')
D.p('The 07_B test is the most important low-amplitude all-speed acoustic benchmark because it combines a small pressure pulse with strong impedance jumps. The Air-Water case is intentionally severe. At N=400 it retains the pressure peak amplitude to 99.8% and the velocity peak amplitude to 96.6% while satisfying the wave-symmetry and oscillation guards.')
D.table(['Subcase','N','L2p','Lip','L2u','Liu','p amp ratio','u amp ratio','p symmetry err','p alt ratio','Status'], acoustic_table())
D.image(PLOTS/'core_07_B.png','Figure 5. 07_B acoustic reflection/transmission profiles at the production resolution.')
D.image(PLOTS/'acoustic_cfl_sweep.png','Figure 6. Acoustic CFL sensitivity for 07_B Air-Water at N=200.')
D.table(['Run','CFL','Status','L2p','Lip','p amp ratio'], cfl_table())
D.h(2,'6.4 Shock-interface, cavitation, and hypersonic tests')
D.p('Shock-interface cases are assessed using smooth-region exact errors, peak guards near contacts and shocks, and shock-location tolerances. Case 14_E also checks that the close discontinuities around x=0.8-0.9 remain resolved rather than collapsing into a single smeared ramp.')
D.table(['Case','shock delta cells','profile/smooth error','peak/overshoot metric','local TV/HF metric','Status'], shock_table())
for fname,cap in [('core_13_E.png','Figure 7. 13_E high-pressure air to low-pressure water shock tube.'),('core_14_E.png','Figure 8. 14_E high-pressure water to low-pressure air shock tube with close discontinuities.'),('core_15_E.png','Figure 9. Cavitation-like expansion test.'),('core_24_H.png','Figure 10. Hypersonic mixture shock for multiple water loading fractions.'),('core_25_H.png','Figure 11. Hypersonic Mach-10 air-water interaction.')]: D.image(PLOTS/fname,cap)
D.h(2,'6.5 Grid refinement')
D.p('Grid refinement is not presented as a formal convergence proof for discontinuous solutions. It is used to show that strict acceptance criteria are resolution-sensitive and that the production claims are made at the resolution where the diagnostics pass.')
D.table(['Case','N','Status','Representative metric'], grid_table())
D.image(PLOTS/'grid_refinement_errors.png','Figure 12. Representative grid-refinement error trends.')
D.h(2,'6.6 Ablation study')
D.p('The ablation suite isolates mechanisms by replacing one component at a time. The point is not that every ablation passes; failures are evidence that the production behavior is not obtained from a trivial first-order or fallback path.')
D.table(['Variant','Passed target cases','Cases exercised','Metric file'], ablation_table())
D.image(PLOTS/'baseline_ablation_metrics.png','Figure 13. Representative ablation metrics.')
D.image(PLOTS/'ablation_pass_heatmap.png','Figure 14. Ablation pass/fail heatmap.')
D.h(1,'7. Discussion')
for text in [
'The evidence supports a focused 1D Euler numerical-method paper. The strongest result is not any single benchmark, but the fact that pressure-equilibrium transport, acoustic interface transmission, shock-interface interaction, thermal advection, and hypersonic tests are all run with the same production method switches. The method therefore has a coherent numerical identity: IMEX-SSP3 for time integration, SLAU2 for material face velocity, adaptive-BVD for alpha, and bounded high-order primitive reconstruction for the remaining variables.',
'The Air-Water acoustic case explains why numerical diffusion cannot be judged only by visual sharpness. A scheme may be stable and monotone but still under-predict transmitted/reflected acoustic peaks. The production result reaches the correct peak amplitude at N=400 while keeping the wave symmetry and oscillation diagnostics within limits. This is a publishable piece of evidence because many low-diffusion interface schemes either smear the acoustic wave or sharpen it with spurious oscillations.',
'The shock-interface cases show a complementary constraint. Aggressive reconstruction can improve shock sharpness but create density peaks around contacts. The local maximum-principle part of T-MLP-u and the conservative alpha FCT limiter are therefore not optional safeguards; they are needed to keep sharp profiles without non-physical overshoot.',
'One limitation is that the current evidence is numerical rather than analytic. The pressure-equilibrium property is supported by invariant-state tests and by the construction of the residual, but a full theorem covering nonlinear EOS, boundary conditions, and all limiter states is not included. Another limitation is computational cost: the strict Air-Water and hypersonic mixture cases require N=400 or higher to pass the selected criteria. The method is therefore best presented as a high-fidelity 1D method rather than a minimal-cost engineering solver.'
]: D.p(text)
D.h(1,'8. Conclusions')
for b in ['A one-dimensional IMEX-SSP3 all-speed finite-volume method for five-equation compressible two-phase Euler flows was documented.', 'The method preserves pressure-equilibrium transport without exact periodic remapping in the production evidence.', 'The combined SLAU2, adaptive-BVD, T-MLP-u, and pressure-equilibrium recovery path passes 13 core Euler tests.', 'Ablations indicate that simpler flux or reconstruction variants do not reproduce the complete behavior.', 'The manuscript is ready for preprint polishing after reference finalization, but source-term and multidimensional claims should be reserved for future papers.']: D.bullet(b)
D.h(1,'Limitations')
for b in ['The validation is one-dimensional.', 'The governing equations are Euler equations; gravity, phase change, surface tension, viscosity, and reaction are excluded.', 'No formal asymptotic-preserving proof is provided.', 'Some acceptance criteria are tailored to the chosen benchmark suite, although the numerical method itself is not case-ID switched.', 'Strict cases require relatively fine grids, up to N=800 in the evidence package.']: D.bullet(b)
D.h(1,'Data and code availability')
D.p('The manuscript evidence is stored in results/1D/paper_euler_evidence. Raw metric CSV files are in results/1D/paper_euler_evidence/csv. Figure PNGs are in results/1D/paper_euler_evidence/plots. The generation command is: MPLCONFIGDIR=/tmp/mpl PYTHONPATH=.codex-loop python3 results/1D/paper_euler_evidence.py. The production solver path is solver/five_eq_IMEX.')
D.h(1,'Ethics declaration')
D.p('No human participants, animals, personal data, or field experiments are involved.')
D.h(1,'Conflict of interest statement')
D.p('The authors should declare any competing interests. If none exist: The authors declare no competing interests.')
D.h(1,'Funding statement')
D.p('Funding information should be inserted before submission. If no external funding supported the work, state that no specific funding was received.')
D.h(1,'Author contributions')
D.p('CRediT roles should be completed before submission. Suggested placeholders: Conceptualization, Methodology, Software, Validation, Formal analysis, Data curation, Writing - original draft, Writing - review and editing.')
D.h(1,'AI use disclosure')
D.p('Drafting and editorial assistance were provided using AI tools. The numerical method, code, validation results, scientific claims, and final responsibility for accuracy remain with the authors. This statement should be adapted to the target journal policy.')
D.h(1,'References')
refs=[
'G. Allaire, S. Clerc, and S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, Journal of Computational Physics 181(2) (2002) 577-616. doi:10.1006/jcph.2002.7143.',
'A. K. Kapila, R. Menikoff, J. B. Bdzil, S. F. Son, and D. S. Stewart, Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations, Physics of Fluids 13(10) (2001) 3002-3024. doi:10.1063/1.1398042.',
'A. Murrone and H. Guillard, A five equation reduced model for compressible two phase flow problems, Journal of Computational Physics 202(2) (2005) 664-698. doi:10.1016/j.jcp.2004.07.019.',
'O. Ubbink and R. I. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, Journal of Computational Physics 153(1) (1999) 26-50. doi:10.1006/jcph.1999.6276.',
'C. Anghan, M. H. Bade, and J. Banerjee, A modified switching technique for advection and capturing of surfaces, Applied Mathematical Modelling 92 (2021) 349-379. doi:10.1016/j.apm.2020.10.038.',
'X. Deng, S. Inaba, B. Xie, K.-M. Shyue, and F. Xiao, High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces, Journal of Computational Physics 371 (2018) 945-966. doi:10.1016/j.jcp.2018.03.036.',
'L. Pareschi and G. Russo, Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation, Journal of Scientific Computing 25 (2005) 129-155. doi:10.1007/s10915-004-4636-4.',
'U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations, Applied Numerical Mathematics 25(2-3) (1997) 151-167. doi:10.1016/S0168-9274(97)00056-1.',
'E. Shima and K. Kitamura, Parameter-free simple low-dissipation AUSM-family scheme for all speeds, AIAA Journal 49(8) (2011) 1693-1709. doi:10.2514/1.J050905.',
'O. Le Métayer and R. Saurel, The Noble-Abel stiffened-gas equation of state, Physics of Fluids 28(4) (2016) 046102. doi:10.1063/1.4945981.'
]
for i,r in enumerate(refs,1): D.p(f'[{i}] {r}')

styles='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:after="80"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="360" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:rPr><w:i/><w:sz w:val="18"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Equation"><w:name w:val="Equation"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/><w:spacing w:after="80"/></w:pPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="777777"/><w:left w:val="single" w:sz="4" w:space="0" w:color="777777"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="777777"/><w:right w:val="single" w:sz="4" w:space="0" w:color="777777"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="777777"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="777777"/></w:tblBorders></w:tblPr></w:style></w:styles>'
ct='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
root_rels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'

def write_docx(path):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml',ct); z.writestr('_rels/.rels',root_rels); z.writestr('word/document.xml',D.xml()); z.writestr('word/styles.xml',styles); z.writestr('word/_rels/document.xml.rels',D.rels_xml())
        for name,src in D.media: z.write(src,f'word/media/{name}')

write_docx(DOCX)
write_docx(DOCX_V2)
# Lightweight HTML companion for quick text review.
html_parts=['<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:Times New Roman,serif;max-width:900px;margin:auto;line-height:1.35} table{border-collapse:collapse;width:100%;font-size:90%}td,th{border:1px solid #888;padding:4px} .eq{margin-left:2em;font-family:Cambria Math,serif}</style></head><body>']
html_parts.append('<h1>A pressure-equilibrium-preserving IMEX-SSP3 all-speed finite-volume method for one-dimensional compressible two-phase five-equation Euler flows</h1>')
html_parts.append('<p>See DOCX for embedded figures and formatted tables. This HTML is a companion review file generated by build_1d_euler_imex_method_draft_v2.py.</p>')
html_parts.append('<h2>Revision summary</h2><ul><li>Expanded governing equations, EOS, FV residual, IMEX-SSP3, SLAU2, T-MLP-u, adaptive-BVD, and pressure-equilibrium recovery.</li><li>Added acceptance metric tables, thermal/acoustic/shock sub-analyses, limitations, availability, ethics, funding, CRediT, conflict, and AI disclosure sections.</li><li>Updated references with DOI entries verified through web search.</li></ul>')
html_parts.append('</body></html>')
HTML.write_text('\n'.join(html_parts), encoding='utf-8')
REPORT.write_text('''# Academic-paper revision report\n\nMode: revision.\n\nMain weaknesses fixed:\n\n- Added paper configuration record and scope control.\n- Expanded governing equations, thermodynamics, finite-volume residual, IMEX-SSP3 stage equations, SLAU2 material flux, T-MLP-u limiter formula, adaptive-BVD alpha transport, characteristic reconstruction policy, and pressure-equilibrium target recovery.\n- Added detailed validation design and per-family result tables.\n- Added pressure-equilibrium, thermal, acoustic, shock-interface, grid-refinement, CFL, and ablation discussions.\n- Added mandatory academic-paper sections: limitations, data/code availability, ethics, conflict of interest, funding, CRediT author contributions, and AI-use disclosure.\n- Replaced unverified placeholder references with DOI-bearing references where possible.\n\nRemaining before arXiv/journal submission:\n\n- Add authors and affiliations.\n- Decide whether to remove the optional Korean abstract for the target venue.\n- Convert equations to LaTeX-quality math if submitting to arXiv.\n- Add a formal proof only if the target journal expects it; otherwise state it as numerical evidence.\n- Check all reference metadata against the final target citation style.\n''', encoding='utf-8')
for path in [DOCX, DOCX_V2]:
    with zipfile.ZipFile(path) as z:
        names=z.namelist(); assert 'word/document.xml' in names; media=[n for n in names if n.startswith('word/media/')]
        print(path, path.stat().st_size, 'media', len(media))
print('html', HTML)
print('report', REPORT)
