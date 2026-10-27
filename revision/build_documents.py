"""Build honest revision drafts from frozen sources. Never insert invented results."""
import difflib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper'
REFERENCE = ROOT/'revision/reference/paper'

# Scores measure the present response/manuscript fix, not predicted acceptance.
CONCERNS = [
dict(id='R1-C1', priority=10, kind='Protocol disclosure; shared experiment', score=7,
 response=r'''The historical study used stratified random sampling of labeled pixel centers within each scene. Patches were constructed before partitioning, so distinct center indices do not imply independent neighborhoods. PCA with whitening was fitted to the complete scene before splitting; this transductive preprocessing is now disclosed. Validation accuracy and loss were monitored, but the inspected training script evaluated the final epoch rather than selecting a validation-best checkpoint. We therefore restrict the interpretation of the historical scores to within-scene classification.''',
 action=r'''Revised the experimental protocol and added an explicit evaluation limitation in Sections III and VI. Section V.E specifies training-only PCA, five pipeline repetitions, and a buffered spatial protocol for the new study. \pending{The five-seed and spatial tables require completed runs; the historical table values are not repeated-run estimates.}''',
 locations=['sec:experiments','sec:revision_protocol','sec:limitations']),
dict(id='R1-C2', priority=10, kind='Architectural correction; controlled experiment', score=8,
 response=r'''The reviewer is correct. The implemented recurrence uses two biased dense maps with parameters shared over token positions and no recurrent activation. It is an affine linear recurrent state-space aggregator, not a selective Mamba block: it does not compute input-dependent state-space parameters. We have made this distinction explicit while retaining FKANMamba as the model identifier. The existing results establish performance of the complete model, not a separate benefit of recurrence.''',
 action=r'''Revised Section II.E and the architectural claims. Added mean-pooling and selective-Mamba aggregation controls to Section V.E. \pending{The aggregation controls have not yet supplied experimental evidence of an independent recurrent-stage benefit.}''',
 locations=['sec:ssm','sec:revision_protocol']),
dict(id='R1-C3', priority=9, kind='Implementation disclosure; unresolved historical provenance', score=6,
 response=r'''Section II.A now gives the exact transpose/reshape operations and coordinate mapping. Fusion pairs equal sequence indices, not necessarily equal physical pixels. The scan visits indices 0 through 143, and FEB uses zero-based index 6 for a 12 by 12 patch. We specify eight RBF centers initialized over [-1,1], trainable grid centers, 32 output channels, eight heads, independent token projections and FEB gates, and shared FSSA weights across the branches. The difference between the historical NA and without-FSSA configurations cannot be established from their table values alone.''',
 action=r'''Revised Sections II.A--II.E and annotated the historical attention/module tables. Defined a reproducible without-FSSA variant that bypasses attention while retaining FEB, fusion, and recurrence. \pending{Recover the original NA configuration or replace the untraceable comparison with new controlled results. No architectural explanation for its different HC score is asserted.}''',
 locations=['sec:token_details','sec:fastkan','sec:feb','sec:fssa','sec:ablation']),
dict(id='R1-C4', priority=9, kind='Claim correction; shared profiling experiment', score=8,
 response=r'''We agree that parameter count is distinct from runtime and memory efficiency. The model has fewer parameters than the listed Transformer/KAN baselines but is not the smallest among the listed Mamba-labeled baselines. The attention stages retain quadratic token interactions; a subsequent linear-length scan does not make the complete model linear in sequence length. The training-ratio figure measures a within-model trade-off rather than a speed comparison between methods.''',
 action=r'''Restricted the efficiency claims in the abstract, results, and conclusion, and added complexity accounting in Section II.E. \pending{The controlled runtime, memory, and FLOPs table requested by R2-C3 remains to be measured.}''',
 locations=['sec:ssm','sec:revision_protocol','sec:conclusion']),
dict(id='R1-C5', priority=5, kind='Numerical correction; figure readability', score=7,
 response=r'''The HH difference is 98.63 - 95.83 = 2.80 percentage points, rather than 1.90 percentage points. The revised discussion uses the correct difference and distinguishes absolute percentage-point gains from relative percentage changes.''',
 action=r'''Corrected the HH discussion. Added enlarged external panel labels to revision-specific classification figures when the panel order can be read from the source figure. \pending{Final numerical and visual checks must also cover any replacement results and classification maps.}''',
 locations=['tab:HH','fig:HC','fig:HH','fig:TH','fig:PI','fig:QUH']),
dict(id='R2-C1', priority=10, kind='Major experimental validity; new protocol', score=4,
 response=r'''The original partition is pixel-center random sampling, not spatially independent sampling. Its overlapping neighborhoods and full-scene PCA can yield optimistic within-scene estimates. The revised manuscript states this limitation. The additional protocol assigns spatial blocks before patch extraction and excludes centers whose full patch support crosses a partition boundary; PCA is fitted only to training-center spectra.''',
 action=r'''Added the spatial protocol and support-disjointness checks in Section V.E, with HC and HH selected in advance for the spatial comparison. \pending{Spatial classification results over all five seed configurations are not yet available. This concern is not considered experimentally resolved.}''',
 locations=['sec:experiments','sec:revision_protocol','sec:limitations']),
dict(id='R2-C2', priority=10, kind='Architectural identity', score=9,
 response=r'''The proposed recurrence is mathematically an affine linear recurrent network: the hidden state and input each undergo a biased dense map, whose outputs are added. It is also a non-selective state-space recurrence with fixed learned matrices. Conventional selective Mamba additionally generates input-dependent selection parameters and uses a specialized scan implementation. Those mechanisms are absent from the submitted recurrence. We no longer claim architectural equivalence.''',
 action=r'''Revised Section II.E to specify both biases, the shared matrices, zero initial state, final-state readout, and the absence of recurrent nonlinearity or selective parameters. Clarified that FKANMamba is the retained model name.''',
 locations=['sec:ssm']),
dict(id='R2-C3', priority=9, kind='Required controlled measurement', score=3,
 response=r'''Parameter count alone does not establish computational efficiency. We now distinguish quadratic attention cost, RBF projection cost, and the affine recurrent scan. The historical custom FLOPs routine omitted important operations, so its totals are not used as evidence of comparative efficiency. The new profiling protocol uses the same device, precision, batch sizes, warm-up, synchronization, and clearly identified implementations.''',
 action=r'''Added the profiling specification in Section V.E. \pending{Populate the comparison with measured FLOPs/operation coverage, batch-1 latency, batch-56 throughput, peak training/inference memory, epoch time, and total training time. No comparative speed claim is made before these measurements exist.}''',
 locations=['sec:ssm','sec:revision_protocol']),
dict(id='R2-C4', priority=9, kind='Required independent repetitions', score=3,
 response=r'''The historical scores are single-run values and do not quantify variability. We have specified five independently seeded pipeline repetitions, pairing each split and initialization configuration across methods. The revised analysis will report mean and sample standard deviation and paired OA differences. With only five pairs, an exact two-sided sign-flip test cannot attain p below 0.0625; we will not interpret this limited power as evidence of equivalence.''',
 action=r'''Specified seed pairs (345,0), (346,1), (347,2), (348,3), and (349,4) for split and training randomness in Section V.E. \pending{Execute all five repetitions and insert the summaries and multiplicity-corrected paired tests. The present historical tables remain single-run observations.}''',
 locations=['sec:revision_protocol']),
dict(id='R2-C5', priority=10, kind='Factual implementation correction', score=9,
 response=r'''The target label corresponds to patch coordinate (6,6) under the implemented offsets -6 through +5. Its row-major pixel-token index would be 78. FEB instead selects index 6 from the transposed/reshaped branch. Moreover, that branch groups entries across spatial coordinates and PCA components. Its selected token is therefore a fixed representative descriptor, not the target-pixel spectrum or a unique geometric center. We preserve this operation and correct its description.''',
 action=r'''Revised Sections II.A and II.C and the architecture caption. Removed the center-pixel interpretation and provided the exact mapping. No change to the submitted forward operation is claimed.''',
 locations=['sec:token_details','sec:feb','fig:architecture']),
dict(id='R2-C6', priority=9, kind='Novelty qualification; causal comparisons', score=3,
 response=r'''The component families are established, so the contribution is their investigated combination rather than invention of FastKAN, attention, or state-space modeling. Removal alone cannot isolate the benefit of an RBF mapping over a similarly sized alternative. We have therefore specified matched-interface linear and GELU-MLP projection controls, an MLP gating control, and pooling/selective-Mamba aggregation controls.''',
 action=r'''Qualified the contribution statement and defined seven controlled variants in Section V.E. Parameter differences will be reported rather than assumed matched. \pending{The new ablation results and selective-Mamba reference verification are required before claiming an advantage of these choices.}''',
 locations=['sec:revision_protocol','sec:conclusion']),
dict(id='R2-C7', priority=9, kind='Generalization protocol; shared spatial study', score=4,
 response=r'''Five benchmark scenes do not by themselves demonstrate cross-scene transfer when each model is trained and tested within the same scene. The additional spatial protocol addresses the requested reduction in training/test spatial correlation. It remains a within-scene spatial holdout, and neither it nor the original study establishes cross-sensor or cross-scene generalization.''',
 action=r'''Added the preselected HC/HH spatial evaluation protocol and qualified the generalization claims in Section VI. \pending{Complete and report the spatial results. Cross-scene transfer remains an explicit limitation rather than a completed experiment.}''',
 locations=['sec:revision_protocol','sec:limitations']),
dict(id='R2-C8', priority=9, kind='Reproducibility and release evidence', score=6,
 response=r'''The revision package records center-index partitions, preprocessing fit state, split/training seeds, model configuration, source hashes, environment, commands, predictions, and run-level metrics. Seed changes are propagated to partition construction as well as training. We also distinguish the recovered local model definition from an authenticated historical experiment snapshot.''',
 action=r'''Added the reproducibility protocol in Section V.E and an isolated experiment runner. The recovered HC model has 52,760 parameters versus the historical table's 37,540; the same 15,220-parameter difference occurs at all five class counts. \pending{Authenticate the original model configuration as well as the baseline implementations before full training, complete the experiment archive, and supply the verified public release URL and final environment record before submission. A public release is not claimed here.}''',
 locations=['sec:revision_protocol']),
dict(id='R2-C9', priority=7, kind='Interpretation; repeated controlled evidence', score=7,
 response=r'''The historical QUH result without FEB has OA 98.46 versus 98.37 and kappa 97.96 versus 97.85, while AA decreases from 96.93 to 96.56. This is a metric-dependent single-run observation, not evidence of a consistent FEB benefit or a proven dataset-specific mechanism. OA is frequency-weighted whereas AA weights classes equally, so they can move differently. The physical cause of this outcome is not established by the existing table.''',
 action=r'''Revised Section V.B to state this exception and remove universal-benefit language. Added a five-repetition QUH without-FEB comparison with per-class analysis. \pending{Use the paired results to determine whether the apparent changes persist; do not assert a causal explanation from the current single-run values.}''',
 locations=['sec:ablation','sec:revision_protocol']),
]


def revised(text):
    return '\\begin{revision}\n'+text.strip()+'\n\\end{revision}\n'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one occurrence of {old[:90]!r}; found {text.count(old)}')
    return text.replace(old, new, 1)


def build_response():
    source = (REFERENCE/'Response_Letter.tex').read_text(encoding='utf-8')
    starts = list(re.finditer(r'\{\\color\{reviewgreen\}\\textbf\{Reviewer', source))
    if len(starts) != 14:
        raise ValueError('Expected exactly 14 original reviewer comments')
    preamble = source[:source.index('\\begin{document}')]
    preamble += r'''
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\newcommand{\pending}[1]{\par\noindent\textcolor{red!70!black}{\textbf{Pending evidence:} #1}\par}
\begin{document}
\noindent\textbf{WORKING REVISION DRAFT -- NOT READY FOR SUBMISSION}

\noindent\textbf{Original Manuscript ID:} Access-2026-42627

\noindent\textbf{Article:} FKANMamba: FastKAN-Enhanced State Space Learning for Spectral-Spatial Hyperspectral Image Classification

Dear Editor and Reviewers,

Thank you for reviewing our manuscript. We provide a point-by-point response below. The revised working manuscript clarifies the implemented architecture and the scope of the historical evaluation. Reviewer comments are reproduced in green and author responses are in black. Changes to the manuscript are marked with a yellow background.

This working draft explicitly identifies experiments and release actions that remain pending. It must not be submitted until those entries have been replaced with verified evidence and final manuscript page numbers. We do not claim that the new experiments have already been completed.

Sincerely,\\
The authors
\newpage
'''
    chunks = [preamble]
    for i, (start, concern) in enumerate(zip(starts, CONCERNS)):
        end = starts[i+1].start() if i+1 < len(starts) else source.index('\\end{document}')
        block = source[start.start():end]
        marker = re.search(r'\\textbf\{Author [Rr]esponse:\}', block)
        # The supplied comments contain blank paragraphs inside \textbf, which
        # is not a long macro and therefore fails LaTeX. Preserve words, normalize space.
        comment = re.sub(r'\s+', ' ', block[:marker.start()].strip())
        comment = comment.replace('12×12', r'$12\times12$')
        chunks += [f'% CONCERN-ID: {concern["id"]}\n', comment+'\n\n',
                   '\\textbf{Author response:}\n'+concern['response']+'\n\n',
                   '\\textbf{Author action:}\n'+concern['action']+'\n\n',
                   '\\vspace{1.5em}\n']
    chunks += ['\\end{document}\n']
    (PAPER/'repose_filled.tex').write_text(''.join(chunks), encoding='utf-8')


def build_manuscript():
    source = (REFERENCE/'Highlight_Revision.tex').read_text(encoding='utf-8')
    # Remove inactive duplicate-method history without changing the frozen source.
    source = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('%'))+'\n'
    # The original loads two competing math font stacks; retain the standard
    # Times stack to avoid the amssymb/newtxmath duplicate-Bbbk build failure.
    source = source.replace('\\usepackage{newtxtext,newtxmath}\n','')
    source = source.replace('\\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}',
                            '\\history{Working revision draft; experimental evidence pending.}')
    source = source.replace('\\doi{00.0000/ACCESS.2026.DOI}', '\\doi{}')
    for old,new in [('‘','`'),('’',"'"),('“','``'),('”',"''"),('–','--'),('—','---')]:
        source=source.replace(old,new)
    source = replace_once(source, '\\usepackage{soul}', r'''\usepackage{soul}
\usepackage{framed}
\newif\ifrevisionhighlights
\revisionhighlightstrue
\definecolor{shadecolor}{RGB}{255,255,160}
\newenvironment{revision}{\ifrevisionhighlights\begin{shaded*}\fi}{\ifrevisionhighlights\end{shaded*}\fi}
\newcommand{\pending}[1]{\textcolor{red!70!black}{\textbf{Pending evidence:} #1}}''')
    # Existing foreground-yellow marks are replaced by readable background revisions.
    source = source.replace(r'\textcolor{yellow}{2.80\%}', '2.80 percentage points')
    pattern = r'\\textcolor\{yellow\}\{\s*The matrices.*?fixed \$d_f\$ and \$d_s\$\}'
    source, count = re.subn(pattern, lambda _: 'The transition and update matrices are shared over token positions.', source, flags=re.S)
    if count != 1:
        raise ValueError('Existing state-space yellow block not found')
    begin = source.index('\\begin{abstract}')+len('\\begin{abstract}')
    end = source.index('\\end{abstract}', begin)
    source = source[:begin]+'\n'+revised(r'''We investigate FKANMamba, a patch-based hyperspectral classifier combining radial-basis-function FastKAN mappings, two-branch attention, and a non-selective affine recurrent state-space aggregator. Independent token projections and context-derived channel gates precede shared nonlinear attention. The model identifier does not imply an implementation of selective Mamba. Historical experiments on five hyperspectral scenes report high within-scene classification accuracy and a compact parameter count relative to the listed Transformer and KAN baselines. Those results use overlapping random-pixel neighborhoods and full-scene PCA; they do not establish spatially independent generalization, repeated-run reliability, or comparative runtime superiority. The revision specifies controlled projection and aggregation ablations, five independently seeded pipeline repetitions, and support-disjoint evaluation on HC and HH. \pending{The new experimental and computational results must be inserted before submission.}''')+source[end:]
    source = replace_once(source, '\\section{Proposed Methodology}', revised(r'''\textbf{Working revision status.} Historical tables are retained as single-run within-scene observations. The five-repetition, spatial, controlled-ablation, and profiling studies described in Section~\ref{sec:revision_protocol} are specified but not yet completed. This manuscript is not submission-ready.''')+'\n\\section{Proposed Methodology}')
    # Contribution statements must not imply invention of established components.
    a = source.index('The main contributions of this work are summarized as follows:')
    b = source.index('\\end{itemize}', a)+len('\\end{itemize}')
    source = source[:a]+revised(r'''The contribution is an investigated combination of established operators: RBF-based channel gating and query/key/value mappings are integrated with two token views and an affine recurrent aggregator. We provide an explicit account of the implemented tensor operations and qualify the evidence to within-scene classification. Controlled comparisons are required to establish whether the RBF mappings and recurrent readout improve on simpler alternatives; their individual superiority is not established by the historical complete-model results.''')+source[b:]
    a = source.index('\\subsection{Token Generation}')
    b = source.index('\\subsection{Fast Kolmogorov-Arnold Network}', a)
    source = source[:a]+r'''\subsection{Token Generation}
\label{sec:token_details}
'''+revised(r'''Let $\mathbf X\in\mathbb R^{B\times p\times p\times C_r}$. The implementation constructs $L=p^2$ tokens per branch using
\begin{equation}
\begin{aligned}
F_{\rm spec}&=\operatorname{reshape}(X,[B,p^2,C_r]),\\
F_{\rm spa}&=\operatorname{reshape}(\operatorname{transpose}(X,[0,2,3,1]),[B,p^2,C_r]).
\end{aligned}
\label{eq:patch_tokens}
\end{equation}
These are different memory-order views, not two pixel-aligned sequences. With zero-based token index $t$ and feature index $j$, put $q=tC_r+j$. Then
\begin{equation}
\begin{aligned}
F_{\rm spec}[b,t,j]&=X[b,\lfloor t/p\rfloor,t\bmod p,j],\\
F_{\rm spa}[b,t,j]&=X[b,q\bmod p,\lfloor q/(C_rp)\rfloor,\lfloor q/p\rfloor\bmod C_r].
\end{aligned}
\end{equation}
The latter groups entries from multiple spatial rows and PCA components. Each sequence is projected through its own biased dense map from $C_r=15$ to $d=32$. The projections do not share weights. At fusion, equal sequence indices are concatenated along the channel axis; equal indices do not imply a common physical pixel. Thus the branches are complementary tensor views, and no pixelwise alignment claim is made.
\begin{equation}
S_{\rm spa}=F_{\rm spa}W_{\rm spa}+b_{\rm spa},\qquad
S_{\rm spec}=F_{\rm spec}W_{\rm spec}+b_{\rm spec}.
\label{eq:tokens}
\end{equation}''')+source[b:]
    source = replace_once(source, '\\label{sec:fastkan}', '\\label{sec:fastkan}\n'+revised(r'''All implemented FastKAN maps use eight Gaussian RBF centers initialized uniformly over $[-1,1]$, with fixed initial spacing $2/7$ in the denominator. The centers themselves are trainable; the initial interval is not a constraint on their subsequent values. Feature-axis LayerNorm precedes the basis expansion. The optional base-update branch is disabled. Spline projection weights use Glorot-uniform initialization and an L2 penalty of 0.001; the constructor's spline initialization-scale argument does not control the active initializer.'''))
    source = replace_once(source, '\\subsection{FastKAN Enhancement Block}', '\\subsection{FastKAN Enhancement Block}\n'+revised(r'''FEB selects zero-based index $l_r=p/2=6$ from $S_{\rm spa}$ for the $12\times12$ setting. The target pixel lies at patch coordinate $(6,6)$ under offsets $-6,\ldots,+5$; its row-major pixel-token index would be 78. The selected transposed-branch token is not that pixel's spectrum. We therefore call it a fixed representative token. Independent FastKAN maps followed by sigmoid generate the two channel gates, which are broadcast over token positions. This description preserves the submitted computation.'''))
    source = replace_once(source, '\\subsection{FastKAN Spectral-Spatial Attention}', '\\subsection{FastKAN Spectral-Spatial Attention}\n'+revised(r'''The same FSSA layer instance is applied to both branches, so its query, key, value, and output-projection weights are shared across branches. The Q, K, and V mappings are mutually distinct. Each produces 32 channels, divided into eight heads of width four; the biased output projection also produces 32 channels. Attention-weight dropout is 0.7 during training. Concatenation therefore gives a fused width of 64.'''))
    a = source.index('\\subsection{Recurrent State-Space Module}')
    b = source.index('\\section{Experimental Settings and Dataset Information}', a)
    source = source[:a]+r'''\subsection{Non-Selective Affine Recurrent State-Space Aggregator}
\label{sec:ssm}
'''+revised(r'''The submitted recurrent module is an affine linear recurrent network with hidden width $d_s=64$. It scans fused indices $t=0,\ldots,143$ in their constructed order, starts at zero, and uses the final hidden state for classification:
\begin{equation}
h_{t+1}=h_tW_{\rm trans}+O_tW_{\rm update}+b_{\rm trans}+b_{\rm update},\qquad h_0=0.
\label{eq:ssm_update}
\end{equation}
Both dense maps include biases. Their parameters are learned but shared over positions, with no recurrent activation, input-dependent selection, or Mamba-specific discretization. Consequently this is a non-selective affine state-space aggregator, not an implementation of selective Mamba. FKANMamba is retained as the framework identifier. The classifier is a biased dense softmax applied directly to $h_L$; the separately declared dense/dropout members are unused in the forward pass. No normalization is applied between this hidden state and the classifier.
\begin{equation}
\widehat y=\operatorname{softmax}(h_LW_{\rm cls}+b_{\rm cls}).
\label{eq:ssm_softmax}
\end{equation}
Training uses categorical cross-entropy plus the regularization losses attached to active spline weights. For fixed widths, the recurrent scan costs $O(L(d_s^2+d_fd_s))$. The two attention calls retain $O(L^2d)$ arithmetic and $O(HL^2)$ attention-score storage, where $H$ is the head count. The shared weights reduce stored parameters but do not remove the second attention call. RBF expansion additionally costs $O(LdM)$ basis operations and $O(LdMd_{\rm out})$ projection arithmetic per mapping. The complete model therefore does not have linear complexity in token length. Parameter count alone does not establish runtime or memory superiority.''')+'\n'+source[b:]
    # Correct the misleading architecture caption in full.
    a = source.index('\\caption{FKANMamba begins')
    b = source.index('\n\\label{fig:architecture}', a)
    source = source[:a]+r'''\caption{\colorbox{yellow}{Revised operation diagram.} FKANMamba uses two tensor-order token views, independent projections and FEB gates, shared FastKAN attention, indexwise concatenation, and a non-selective affine recurrent aggregator. FEB uses a fixed representative token, not the target-pixel spectrum. A dense softmax classifier reads the final hidden state; no post-recurrence normalization is used.}'''+source[b:]
    source = source.replace('{Fig/FKANMamba.png}', '{revision_assets/figures/architecture.png}')
    # Replace the historical protocol paragraph, retaining separate dataset descriptions.
    a = source.index('All models are trained using the same data split')
    b = source.index('\\section{Results Analysis}', a)
    source = source[:a]+revised(r'''\textbf{Historical protocol.} The inspected scripts construct patches for labeled centers, exclude label zero, and stratify two random splits by center label. A 90\% test fraction followed by a 50/50 division of the remaining centers yields approximately 5\% training and 5\% validation; the historical ratio arguments are relative weights within the non-test remainder. The default split seed is 345 and is not connected to the training-seed argument. Adjacent sets can share patch pixels. Zero padding is used at scene boundaries. PCA is fitted to the flattened full scene with whitening, retaining 15 components; no separate standard-scaling step is present in this path. This exposes held-out scene spectra to preprocessing and makes the historical evaluation transductive. The historical results are therefore within-scene single-run observations, not spatially independent or cross-scene validation.

FKANMamba uses 50 epochs, batch size 56, legacy Adam with learning rate $10^{-4}$ and optimizer learning-rate decay $10^{-6}$, gradient clipping at norm 1.0, embedding width 32, eight heads, and hidden width 64. The optimizer decay is not weight decay; active spline weights separately carry an L2 penalty. Validation loss and accuracy are monitored, but the inspected path evaluates the final epoch without validation-based checkpoint selection. The current local main script contains committed merge-conflict markers; its recovered incoming-side definition is a candidate reference whose relation to every historical run requires verification. The original baseline partitions, initializations, implementations, and environment cannot all be authenticated from aggregate table values. Thus the earlier assertion that all methods used identical partitions is unverified for the archived results. The new protocol below enforces shared partitions and records their hashes.''')+source[b:]
    # Historical result prose: keep values in the tables; avoid unsupported causality/map claims.
    source = source.replace('To ensure a consistent comparison protocol, all methods were evaluated using the same training, validation, and test partitions and the same patch size.', 'The archived comparisons are reported under the historical protocol; their exact per-method provenance remains to be authenticated.')
    for dataset, label in [('HC','HC'),('HH','HH'),('TH','TH'),('PI','PI'),('QUH','QUH')]:
        patterns = {
          'HC':r'The results on the HC dataset are reported.*?(?=\\begin\{figure\})',
          'HH':r'Table~\\ref\{tab:HH\} reports the results.*?(?=\\begin\{figure\})',
          'TH':r'The TH results are shown.*?(?=\\input\{tables/th_acc\})',
          'PI':r'The PI results are summarized.*?(?=\\input\{tables/pi_acc\})',
          'QUH':r'The QUH results are reported.*?(?=\\input\{tables/quh_acc\})'}
        text = f'Table~\\ref{{tab:{label}}} and Fig.~\\ref{{fig:{label}}} show the historical {dataset} within-scene results. These scores are single-run observations under overlapping random-pixel sampling and full-scene PCA. They establish neither repeated-run superiority nor spatially independent generalization.'
        if dataset == 'HH':
            text += r' The reported OA difference between FKANMamba and SSF is $98.63-95.83=2.80$ percentage points. SSFTTKAN has higher AA, so the table does not show uniform superiority across metrics.'
        text += ' Parameter counts measure storage size; the maps are qualitative illustrations rather than independent evidence of deployment performance.'
        source, n = re.subn(patterns[dataset], lambda _: revised(text)+'\n', source, count=1, flags=re.S)
        if n != 1:
            raise ValueError('Missing result paragraph '+dataset)
    # Replace unsupported ablation causal explanations while preserving the historical tables.
    a = source.index('Table~\\ref{tab:attention_comparison} compares')
    b = source.index('\\input{tables/diff_atn}', a)
    source = source[:a]+revised(r'''Table~\ref{tab:attention_comparison} contains historical single-run comparisons. Its NA row cannot be assumed identical to the without-FSSA row in Table~\ref{tab:feb_fssa}; their configurations must be recovered before a causal comparison is defensible. The observed attention differences alone do not isolate nonlinear projection capacity, normalization, parameter count, or optimization. New controlled variants are specified in Section~\ref{sec:revision_protocol}.''')+source[b:]
    a = source.index('Table~\\ref{tab:feb_fssa} shows')
    b = source.index('\\input{tables/feb_fssa_abliation}', a)
    source = source[:a]+revised(r'''The historical module table shows dataset- and metric-dependent effects. On QUH, without FEB has OA 98.46\% versus 98.37\% and $\kappa$ 97.96\% versus 97.85\%, while AA falls from 96.93\% to 96.56\%. OA weights classes by frequency whereas AA averages class accuracies equally, so changes need not have the same direction. These small single-run differences do not establish a reliable FEB benefit or harm. No physical mechanism is inferred without supporting analysis. Larger drops after FSSA removal indicate an association for that full variant, not proof that RBF projections alone cause the difference. The new study pairs repetitions and reports per-class outcomes.''')+source[b:]
    # Remove claims that patch or training ratio was validated using the test set.
    a = source.index('Table~\\ref{tab:patch_acc} shows')
    b = source.index('\\subsection{Training-Ratio', a)
    source = source[:a]+revised(r'''Table~\ref{tab:patch_acc} is a historical sensitivity analysis under overlapping patch sampling. Increasing patch size changes both available context and train/test neighborhood overlap, so its accuracy trend cannot be attributed solely to richer context. The new protocol fixes $p=12$ in advance and does not select a patch size using held-out test scores.''')+source[b:]
    a = source.index('Fig.~\\ref{fig:diff_ratio} compares')
    b = source.index('\\section{Conclusion}', a)
    source = source[:a]+revised(r'''Fig.~\ref{fig:diff_ratio} reports historical accuracy and training cost as the labeled ratio changes within FKANMamba. It is not an inter-model runtime benchmark. The revision fixes the labeled ratio in advance rather than selecting it from held-out test accuracy. The figure does not demonstrate computational superiority over other methods.''')+'\n'+revision_protocol()+'\n'+source[b:]
    a = source.index('This work presented FKANMamba', source.index('\\section{Conclusion}'))
    b = source.index('\\section*{Acknowledgment}', a)
    source = source[:a]+revised(r'''FKANMamba combines established RBF mappings, two-view attention, and a non-selective affine recurrent readout. The corrected description preserves the implemented operations while separating them from selective Mamba. Historical single-run within-scene results suggest a useful compact classifier, but they do not establish repeated-run superiority, spatially independent generalization, or comparative runtime advantage. The controlled experiments in Section~\ref{sec:revision_protocol} are required before drawing stronger conclusions. \pending{Replace this provisional conclusion with claims supported by the completed revision study.}

\textbf{Limitations.}\label{sec:limitations} Historical spatial overlap and transductive preprocessing limit the interpretation of accuracy. The fixed representative token is not the target spectrum, the two views are not pixel-aligned, and the recurrence is non-selective. The planned spatial comparison covers HC and HH only; it cannot establish transfer between scenes or sensors. Five pipeline repetitions measure variability conditional on these datasets and protocols, with limited power for paired tests. Runtime depends on implementation and hardware, especially for a sequential TensorFlow selective-scan reference. Cross-scene transfer and deployment validation remain outside the present evidence.''')+source[b:]
    # Isolate all historical table inputs from the originals.
    tables = PAPER/'revision_assets/tables'
    tables.mkdir(parents=True, exist_ok=True)
    for path in (REFERENCE/'tables').glob('*.tex'):
        text = path.read_text(encoding='utf-8')
        if path.name != 'datasets.tex':
            # Wrap each caption with a candid historical marker.
            text = re.sub(r'\\caption\{([^{}]*)\}',
                          lambda m: r'\caption{Historical single-run within-scene results. '+m.group(1)+'}', text, count=1)
        (tables/path.name).write_text(text, encoding='utf-8')
    source = source.replace('\\input{tables/', '\\input{revision_assets/tables/')
    for dataset in ('HC','HH','PI','TH','QUH'):
        pattern = r'\\begin\{figure\}(?:\[[^\]\r\n]*\])?\s*\\centering\s*\\includegraphics\[width=\\columnwidth\]\{Fig/'+dataset+r'\.png\}\s*\\caption\{[^{}]*\}\s*\\label\{fig:'+dataset+r'\}\s*\\end\{figure\}'
        replacement = (r'\begin{figure*}[t]'+'\n'+r'\centering'+'\n'
                       +r'\includegraphics[width=0.98\textwidth]{revision_assets/figures/'+dataset+r'.png}'+'\n'
                       +r'\caption{Historical within-scene classification maps on '+dataset+r'. External panel labels and layout have been revised for readability; predictions and class colors are unchanged.}'+'\n'
                       +r'\label{fig:'+dataset+r'}'+'\n'+r'\end{figure*}')
        source, count = re.subn(pattern,lambda _:replacement,source,flags=re.S)
        if count != 1:
            raise ValueError('Expected one map figure for '+dataset)
    support=PAPER/'revision_assets/support'
    (support/'bib').mkdir(parents=True,exist_ok=True)
    for filename in ('IEEEtran.cls','logo.png','notaglinelogo.png','bullet.png'):
        shutil.copyfile(PAPER/filename,support/filename)
    for path in (PAPER/'bib').iterdir():
        if path.is_file():
            shutil.copyfile(path,support/'bib'/path.name)
    bibliography=(REFERENCE/'sample.bib').read_text(encoding='utf-8')
    for old,new in [('‘','`'),('’',"'"),('“','``'),('”',"''"),('–','--'),('—','---')]:
        bibliography=bibliography.replace(old,new)
    (support/'sample.bib').write_text(bibliography,encoding='utf-8')
    shutil.copyfile(PAPER/'Fig/fig_training_time_ratio.png',PAPER/'revision_assets/figures/training_ratio.png')
    source=source.replace('{Fig/fig_training_time_ratio.png}','{revision_assets/figures/training_ratio.png}')
    source=source.replace('{bib/','{revision_assets/support/bib/')
    source = source.replace('\\bibliography{IEEEabrv,sample}', '\\bibliography{revision_assets/support/sample}')
    # The supplied 2014 class uses pdfTeX-only spot-color primitives even under
    # XeTeX. Keep its pdfTeX behavior and use a CMYK fallback for draft QA.
    cls = (PAPER/'ieeeaccess.cls').read_text(encoding='utf-8')
    start = cls.index('  \\RequirePackage{spotcolor}')
    stop = cls.index('  \\definecolor{greycolor}',start)
    spot = cls[start:stop]
    cls = cls[:start]+'\\ifpdf\n'+spot+'\\else\n  \\definecolor{accessblue}{cmyk}{1,0.3,0,0.2}\n\\fi\n'+cls[stop:]
    cls=cls.replace('\\LoadClass{IEEEtran}','\\LoadClass{revision_assets/support/IEEEtran}')
    cls=cls.replace('{notaglineLogo.png}','{revision_assets/support/notaglinelogo.png}')
    cls=cls.replace('{Logo.png}','{revision_assets/support/logo.png}')
    cls=cls.replace('{bullet.png}','{revision_assets/support/bullet.png}')
    cls=cls.replace('VOLUME\\ \\thevol, 2016','WORKING REVISION')
    cls=cls.replace('VOLUME\\ \\thevol, \\theyear','WORKING REVISION')
    (PAPER/'revision_assets/ieeeaccess_revision.cls').write_text(cls,encoding='utf-8')
    source = replace_once(source,'\\documentclass{ieeeaccess}', '\\documentclass{revision_assets/ieeeaccess_revision}')
    source = replace_once(source,'\\end{document}', '\\EOD\n\\end{document}')
    source=source.replace('This description preserves the submitted computation.',
                          'This description preserves the inspected local computation; identity with the historical run configuration requires verification.')
    source=source.replace('FKANMamba uses 50 epochs, batch size 56,',
                          'The recovered configuration uses 50 epochs, batch size 56,')
    source=source.replace('whose relation to every historical run requires verification.',
                          'whose relation to every historical run requires verification. Its HC parameter count is 52,760, whereas the historical table reports 37,540; every dataset shows the same 15,220-parameter discrepancy. Full revision training is blocked until this configuration mismatch is resolved, and the archived parameter totals are not attributed to the recovered model.')
    (PAPER/'highlighted_filled.tex').write_text(source, encoding='utf-8')
    clean = source.replace('\\revisionhighlightstrue', '\\revisionhighlightsfalse', 1)
    clean = clean.replace('\\colorbox{yellow}{Revised operation diagram.}', 'Revised operation diagram.')
    (PAPER/'clean_filled.tex').write_text(clean, encoding='utf-8')
    # Save source-level changes, including deletions that cannot be highlighted.
    initial = (REFERENCE/'Intial_Submission.tex').read_text(encoding='utf-8')
    diff = difflib.unified_diff(initial.splitlines(True), source.splitlines(True),
                                fromfile='Intial_Submission.tex', tofile='highlighted_filled.tex')
    (PAPER/'revision_changes.diff').write_text(''.join(diff), encoding='utf-8')


def revision_protocol():
    return r'''\subsection{Revision Study Protocol and Evidence Status}
\label{sec:revision_protocol}
'''+revised(r'''\textbf{Status.} This subsection specifies the revision study; it does not report completed new results. The predeclared matrix has 460 runs: nine methods on five scenes under random sampling, nine methods on HC/HH under spatial sampling, seven controlled variants on HC/HH under both protocols, and a QUH without-FEB check, each with five repetitions. HC/HH are chosen before inspecting new scores because of their fine-grained agricultural categories. All completed scheduled runs, including unfavorable results, will be reported.

\textbf{Sampling and preprocessing.} Repetition $r=0,\ldots,4$ uses split seed $345+r$ and training seed $r$. All methods share the center indices for a dataset, protocol, and repetition. Random sampling is class-stratified 5\%/5\%/90\%. The spatial protocol uses $64\times64$ blocks on HC and $24\times24$ blocks on HH. Label-only feasibility checks found that HH could not satisfy the class-coverage and sample-count constraints at block widths 64, 48, or 32; width 24 was fixed before model training or inspection of accuracy. Blocks are assigned using ground-truth class coverage and count constraints, never spectra or accuracy. Approximately 20\% of eligible blocks are allocated to each of training and validation, with the remainder for testing. Training and validation centers are sampled to target 5\% of each scene's labeled class population where feasible. A center is retained only if its entire $12\times12$ patch fits in its own block. Realized fractions, class counts, excluded pixels, and block masks will be reported; the two scenes' block scales and retained test coverage differ. Infeasible class coverage stops the protocol; classes are not silently removed. Train/validation/test patch-support intersections must be zero. PCA with whitening is fitted only to training-center spectra, then applied unchanged to the scene; no additional spectral standardization is introduced. Full-scene preprocessing is confined to the explicitly labeled historical results.

\textbf{Training and reproducibility.} The FKANMamba schedule remains 50 epochs, batch size 56, and the optimizer settings stated above. The final epoch is evaluated; validation is monitored and test scores are not used for configuration selection. Published baseline settings and any justified deviations require an authenticated implementation record. Each run stores source hashes, exact center indices, block assignments, PCA state, seeds, environment, configuration, training history, checkpoint, predictions, and per-class confusion counts. Five repetitions vary both partitioning and training randomness; they are not five initializations on one fixed split. Missing or failed runs are listed and not silently omitted from summaries.

\textbf{Controlled variants.} The seven variants replace FastKAN Q/K/V with normalized linear maps; replace them with normalized GELU MLPs of width 128; replace FEB's FastKAN gates with corresponding MLP gates; bypass FEB; bypass FSSA while keeping FEB/fusion/recurrence; replace recurrence with mean pooling; or replace recurrence with a selective-Mamba block. Interfaces remain fixed and parameter differences are reported. The selective control has model width 64, state width 64, convolution width 4, expansion 2, and a final-position readout. The TensorFlow selective-scan reference is sequential and is not an optimized CUDA implementation; external reference parity is required before attributing results to a verified conventional Mamba block. These controls address projection capacity and aggregation choices separately, not a newly invented component family.

\textbf{Statistics.} OA, AA, $\kappa$, and per-class accuracy will be reported as mean and sample standard deviation over all five repetitions, together with paired OA differences. Primary OA comparisons use two-sided exact paired sign-flip tests with Holm correction within the random and spatial comparison families. With five pairs the minimum attainable two-sided p-value is 0.0625; lack of significance is not equivalence. These repetitions are conditional on the five scenes and do not establish population-wide or cross-scene generalization.

\textbf{Computational measurements.} Record GPU model, driver, CUDA/cuDNN, framework, CPU, host RAM, precision, batch size, and compilation settings. On an otherwise idle GPU, use FP32, warm-up, synchronized repetitions, batch-1 latency, and batch-56 throughput. Report peak training/inference device memory, first-epoch/steady-state epoch time, and total training time, separating preprocessing and initialization. FLOPs accounting includes both attention calls, RBF projections, and repeated scan operations; state the operation convention and unsupported profiler operations. No comparative efficiency claim is made from partial FLOPs alone. Training-only pilots estimate the budget before test evaluation.

\pending{Insert the five-seed main results, HC/HH spatial results, controlled ablations, QUH FEB analysis, paired statistics, and controlled computational table. The exact public archive URL and final PDF page references remain to be supplied.}''')


def build_ledger():
    data = [{k:v for k,v in concern.items() if k not in ('response','action')}
            for concern in CONCERNS]
    (ROOT/'revision/concerns.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    build_response()
    build_manuscript()
    build_ledger()
    print('Built response, highlighted and clean manuscript drafts, diff, and concern ledger.')
