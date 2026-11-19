"""Builds main_revised.tex (changes in blue via \\rev{}) from the submitted main_twocolumn.tex.
Clean version: compile main_clean.tex, which defines \\cleanversion and inputs main_revised.tex."""
import re, sys
src = open('main_twocolumn.tex').read()
n_ok = 0


def rep(old, new, count=1):
    global src, n_ok
    if src.count(old) != count:
        sys.exit(f'PATTERN COUNT {src.count(old)} != {count}:\n{old[:120]}')
    src = src.replace(old, new); n_ok += 1


R = lambda t: '\\rev{' + t + '}'

# ---- preamble: revision markup ------------------------------------------------------------
rep('\\newcommand{\\pmv}[2]{#1{\\scriptsize$\\pm$#2}}',
    '\\newcommand{\\pmv}[2]{#1{\\scriptsize$\\pm$#2}}\n\\usepackage{xcolor}\n'
    '% Revised text is blue. For the clean version, compile main_clean.tex (defines \\cleanversion).\n'
    '\\ifdefined\\cleanversion\\newcommand{\\rev}[1]{#1}\\else\\newcommand{\\rev}[1]{{\\color{blue}#1}}\\fi\n'
    '\\newcommand{\\pms}[2]{#1{\\tiny$\\pm$#2}}')

# ---- abstract -------------------------------------------------------------------------------
rep('Its mask-aware aggregation compares sparsified updates only on the coordinates they carry, so that honest nodes are not mistaken for attackers.',
    R('Its mask-aware aggregation (EASFL-M) compares sparsified updates only on the coordinates they carry, so that honest nodes are not mistaken for attackers.'))
rep('On real readings and the real layout of the 50-mote Intel Lab deployment with injected faults, EASFL keeps every node alive to round 200,',
    R('In a semi-synthetic validation that uses the real readings and layout of the 50-mote Intel Lab deployment with injected faults,') + ' EASFL keeps every node alive to round 200,')
rep('and raises network-level F1 from 0.32 to 0.57.\n\\end{abstract}',
    'and raises network-level F1 from 0.32 to 0.57. ' + R('Against baselines given the same compression and participation, mask-aware aggregation resists sign-flip poisoning significantly better on both data sets.') + '\n\\end{abstract}')

# ---- contributions ----------------------------------------------------------------------------
rep('\\item A two-tier robust aggregation scheme with reputation-based quarantine that requires no trusted root dataset, including a mask-aware direction filter and trimmed mean that make Byzantine-robust aggregation work on top-$k$ sparsified updates, and a zero-communication on-node personalization step.',
    '\\item ' + R('A mask-aware two-tier robust aggregation rule (EASFL-M) with reputation-based quarantine that requires no trusted root dataset. Its direction filter and trimmed mean act only on the coordinates each top-$k$ sparsified update carries, so Byzantine-robust aggregation works on compressed updates without discarding honest ones.') + ' A zero-communication on-node personalization step completes the framework.')
rep('ablations, four attack models against three robust-aggregation baselines,',
    'ablations, four attack models against ' + R('five robust-aggregation baselines that use the same compression, participation and cluster-head policy as EASFL (20 seeds, paired significance tests), detection quality over the whole network lifetime,'))

# ---- III-A definitions --------------------------------------------------------------------------
rep('A node dies when its energy reaches zero. Lifetime is reported as the rounds until the first node dies (FND), half the nodes die (HND) and 90\\% die (LND).',
    'A node dies when its energy reaches zero. ' + R('Lifetime is reported as the number of rounds until the first node dies (FND), half of the nodes have died (HND) and 90\\% have died (LND). $E_{200}$ denotes the total energy consumed by all nodes during the first 200 rounds, the length of the learning experiments.'))

# ---- IV-C quantization ------------------------------------------------------------------------
rep('sends the result $q_i$ and sets $r_i\\leftarrow v_i-q_i$.',
    'sends the result $q_i$ and sets $r_i\\leftarrow v_i-q_i$. ' + R('Each kept entry is rounded to the nearest level, $q_{i,k}=s\\,\\mathrm{round}(v_{i,k}/s)$ with $s=\\max_k|v_{i,k}|/127+10^{-12}$; indices are sent as 9-bit integers and $s$ as one 32-bit float.'))

# ---- IV-D mask-aware paragraph ---------------------------------------------------------------------
i0 = src.index('\\textit{Mask-aware variant (EASFL-M).}')
i1 = src.index('\\subsection{On-Node Personalization}')
src = src[:i0] + R(r'''\textit{Mask-aware aggregation (EASFL-M, the proposed rule).} Both the direction test and the trimmed mean above treat the zeros left by top-$k$ sparsification as values. Two honest nodes that keep different coordinates then appear to disagree, and a coordinate kept by only a few nodes is pulled towards zero by the others. EASFL-M makes both steps aware of each update's support $\mathcal{S}_i=\{j:u_{i,j}\neq0\}$. Let $c_j$ be the number of received updates that kept coordinate $j$. The reference is $\tilde u_j=\mathrm{med}\{u_{i,j}:j\in\mathcal{S}_i\}$ if $c_j\ge1$ and $\tilde u_j=0$ otherwise, and the direction test compares $u_i$ with $\tilde u$ on $\mathcal{S}_i$ only:''') + r'''
\begin{equation}
\cos_{\mathcal{S}_i}(u_i,\tilde u)=\frac{\sum_{j\in\mathcal{S}_i}u_{i,j}\tilde u_j}{\|u_i\|\,\|\tilde u_{\mathcal{S}_i}\|+\epsilon}<0,\qquad \epsilon=10^{-12}.
\end{equation}
''' + R(r'''Because $u_i$ is zero outside $\mathcal{S}_i$, $\|u_i\|$ is also its norm on its support. If $\tilde u_{\mathcal{S}_i}=0$ or $\mathcal{S}_i=\emptyset$, the cosine is 0, the direction test does not fire and $u_i$ is judged by the norm test alone. The direction test is applied only when at least three updates reach the CH. Unflagged updates $U'$ are combined per coordinate as $\bar u_j=(c'_j/|U'|)\,\mathrm{TM}_{\beta_j}\{u_{i,j}:i\in U',\,j\in\mathcal{S}_i\}$, where $c'_j$ counts the unflagged updates that kept $j$, $\mathrm{TM}_\beta$ removes the $\lfloor\beta c'_j\rfloor$ smallest and largest values and averages the rest, $\beta_j=0.2$ if $c'_j\ge5$ and 0 otherwise, and $\bar u_j=0$ if $c'_j=0$. Without outliers this equals the plain mean of the sparse updates. If every update at a CH is flagged, none is discarded in that round, but the flags still raise the suspicion scores. A sign-flipped or label-flipped update still opposes the reference on its own support and is flagged, while an honest update with a different support is not. EASFL-M needs no extra communication and the same $O(|U|P\log|U|)$ computation at the CH. Unless stated otherwise, ``EASFL'' in Sections~\ref{sec:matched}--\ref{sec:lifetimef1} denotes EASFL-M; the variant with the dense filter is reported as an ablation.''') + '\n\n' + src[i1:]

# ---- V metrics ------------------------------------------------------------------------------
rep('Because a depleted node cannot raise alarms, we also report \\emph{network-level} F1 at round 200, where every window of a dead node is scored as normal.',
    R(r'''Because a depleted node cannot raise alarms, we also report \emph{network-level} F1. Let $\mathcal{A}(t)$ be the nodes alive at round $t$; the personalized decision $\hat y_w(t)$ on window $w$ of node $i$ is kept if $i\in\mathcal{A}(t)$ and set to 0 (normal) otherwise. True positives, false positives and false negatives are pooled over all test windows of all nodes, $\mathrm{F1}^{\mathrm{net}}(t)=2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$, so every anomaly on a dead node counts as a false negative. We report it at round 200 and, in Section~\ref{sec:lifetimef1}, over the whole lifetime, together with its two components: coverage $|\mathcal{A}(t)|/N$ and live-node F1 (the same pooling over windows of alive nodes only). The synthetic test set has 10{,}000 windows per seed, 900 of them anomalous on average (821--974); the Intel Lab test set has 5{,}000, 461 anomalous on average (424--502).'''))

# ---- Table II local-only network F1 ------------------------------------------------------------------
rep('Local only & \\pmv{0.693}{0.031} & 0.887 & 0.030 & \\pmv{0.708}{0.032} & 0.019 & -- & 0 & -- & 100 & --\\\\',
    'Local only & \\pmv{0.693}{0.031} & 0.887 & 0.030 & \\pmv{0.708}{0.032} & 0.019 & ' + R('\\pmv{0.708}{0.032}') + ' & 0 & -- & 100 & --\\\\')

# ---- Table V caption: dashes ----------------------------------------------------------------------
rep('\\caption{Intel Lab Deployment (50 Motes, Real Readings and Layout, Injected Faults; Mean of 5 Seeds)}\\label{tab:intel}',
    '\\caption{Intel Lab Deployment (50 Motes, Real Readings and Layout, Injected Faults; Mean of 5 Seeds). ' + R('--: Not Applicable or Not Measured in This Run') + '}\\label{tab:intel}')

# ---- new subsections after the mask-aware results ---------------------------------------------------
NEW = r'''
\subsection{Comparison With Matched Baselines}\label{sec:matched}
''' + R(r'''Table~\ref{tab:intel} and Table~\ref{tab:mask} compare EASFL with full-precision FedAvg, so part of each difference comes from compression and participation. To isolate the aggregation rule, we give every baseline EASFL's own top-10\% 8-bit compression with error feedback, energy-aware participant selection ($\rho=0.5$), energy-aware CH selection, 8-bit CH-to-BS and 16-bit broadcast. Only the aggregation differs: mean (FedAvg), coordinate median, trimmed mean (0.2), Krum, a support-aware median without filter or quarantine, EASFL with the dense filter, and EASFL-M. Traffic and energy are then the same for all rules (15.3\,Mbit and 15.4\,J over 200 rounds on the synthetic workload; 7.2\,Mbit and 6.4\,J on Intel Lab). Table~\ref{tab:matched} reports 20 seeds. Differences are tested on paired seeds with the Wilcoxon signed-rank test and Holm correction over the six comparisons in each setting; we also give 95\% bootstrap confidence intervals of the paired difference.

Without attack, EASFL-M is statistically equal to matched FedAvg and to the support-aware median on both data sets (differences 0.000--0.007). Coordinate median and Krum lose 0.12--0.29, because on sparse updates they count the zeros left by sparsification as votes. On Intel Lab the mask-aware rule improves on the dense filter by 0.038 (95\% CI [0.027, 0.048], $p<0.001$), confirming that the dense filter misreads honest heterogeneous updates. Under 20\% sign-flip attackers, EASFL-M is better than every matched baseline on both data sets: by 0.06--0.43 on the synthetic workload (Holm $p\le0.04$) and by 0.15--0.35 on Intel Lab (Holm $p\le0.003$). Under label flip it is better than mean, median and Krum, and equal to trimmed mean on Intel Lab. Under stealth and Gaussian attacks it equals the best matched baseline. With 30\% attackers (synthetic, 5 seeds) no rule is reliable; EASFL-M scores 0.45--0.69 with standard deviations up to 0.25, so we claim robustness only up to 20\%. The energy and lifetime gains of EASFL therefore come from compression, participation and CH selection, which a FedAvg baseline with the same settings shares; the aggregation rule contributes robustness under attack and on heterogeneous real data, not accuracy without attack. The 20-seed means are lower than the 5-seed values of Table~\ref{tab:mask} in some attack settings (e.g. synthetic sign flip $0.64\pm0.12$), which reflects seed-to-seed variability of the attacker set.''') + r'''

\begin{table*}[t]
\caption{''' + R('Global F1 With Matched Baselines: Same Compression, Participation and CH Policy for All Rules, 20\\% Malicious Nodes in Attack Columns (Mean $\\pm$ Std Over 20 Seeds)') + r'''}\label{tab:matched}
\centering\scriptsize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lccccc|ccccc@{}}
\toprule
 & \multicolumn{5}{c|}{Synthetic (100 nodes)} & \multicolumn{5}{c}{Intel Lab (50 motes)}\\
Aggregation & Clean & Sign flip & Stealth & Label flip & Gauss & Clean & Sign flip & Stealth & Label flip & Gauss\\
\midrule
Mean (FedAvg) & \pms{0.674}{0.041} & \pms{0.207}{0.086} & \pms{0.647}{0.068} & \pms{0.403}{0.056} & \pms{0.676}{0.046} & \pms{0.569}{0.035} & \pms{0.141}{0.061} & \pms{0.457}{0.072} & \pms{0.379}{0.102} & \pms{0.550}{0.036}\\
Coord.\ median & \pms{0.564}{0.060} & \pms{0.532}{0.069} & \pms{0.535}{0.064} & \pms{0.534}{0.068} & \pms{0.545}{0.067} & \pms{0.283}{0.097} & \pms{0.201}{0.113} & \pms{0.196}{0.104} & \pms{0.332}{0.069} & \pms{0.160}{0.094}\\
Trimmed mean & \pms{0.669}{0.069} & \pms{0.578}{0.127} & \pms{0.645}{0.065} & \pms{0.540}{0.101} & \pms{0.666}{0.062} & \pms{0.539}{0.029} & \pms{0.246}{0.128} & \pms{0.462}{0.055} & \pms{0.438}{0.085} & \pms{0.519}{0.029}\\
Krum & \pms{0.514}{0.106} & \pms{0.489}{0.122} & \pms{0.449}{0.111} & \pms{0.466}{0.122} & \pms{0.519}{0.115} & \pms{0.305}{0.087} & \pms{0.338}{0.096} & \pms{0.289}{0.104} & \pms{0.380}{0.085} & \pms{0.312}{0.102}\\
Support-aware median & \pms{0.676}{0.062} & \pms{0.572}{0.119} & \pms{0.664}{0.066} & \pms{0.537}{0.093} & \pms{0.685}{0.060} & \pms{0.570}{0.031} & \pms{0.271}{0.092} & \pms{0.467}{0.049} & \pms{0.435}{0.088} & \pms{0.548}{0.031}\\
EASFL, dense filter & \pms{0.672}{0.059} & \pms{0.614}{0.124} & \pms{0.647}{0.086} & \pms{0.606}{0.131} & \pms{0.657}{0.077} & \pms{0.532}{0.030} & \pms{0.422}{0.173} & \pms{0.470}{0.069} & \pms{0.483}{0.086} & \pms{0.513}{0.044}\\
\textbf{EASFL-M} & \pms{0.680}{0.059} & \pms{\textbf{0.639}}{0.118} & \pms{0.662}{0.063} & \pms{\textbf{0.610}}{0.113} & \pms{0.679}{0.053} & \pms{0.570}{0.032} & \pms{\textbf{0.486}}{0.156} & \pms{0.467}{0.057} & \pms{0.463}{0.152} & \pms{0.551}{0.032}\\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Detection Over the Whole Network Lifetime}\label{sec:lifetimef1}
''' + R(r'''The results above stop learning at round 200, while the lifetime runs continue until the nodes are depleted. To show detection quality during the whole survival period, we continue learning every round until the network is dead and evaluate every 25 rounds (synthetic workload, 5 seeds; Table~\ref{tab:lifef1}). We include local-only learning, in which each node trains on its own data every round with the same per-round training energy as an FL participant and reports each flagged window to the BS in one packet. While nodes are alive, the live-node F1 of EASFL-M stays between 0.827 and 0.838 from round 200 to 800, so detection quality does not degrade; network-level F1 then falls only with coverage. Local-only learning detects less well (0.708 at round 200) but, because it never transmits model updates, it keeps 72\% of nodes alive at round 1000. Over rounds 1--500 EASFL-M has the highest cumulative network-level F1 (0.824 against 0.695 for local-only), but over rounds 1--1000 local-only is higher (0.631 against 0.551). This trade-off disappears when FL is used as a training phase: if all schemes train for 200 rounds and afterwards only detect and report alarms (the same accounting for every scheme), EASFL-M leads local-only at every horizon we evaluated (0.812 against 0.687 at round 500, 0.762 against 0.639 at round 1000 and 0.622 against 0.561 at round 2000). Matched FedAvg has the same lifetime curve as EASFL-M, consistent with Section~\ref{sec:matched}.''') + r'''

\begin{table}[t]
\caption{''' + R('Network-Level F1, Coverage and Cumulative Detection Over the Whole Lifetime With Learning Every Round (Synthetic, Mean of 5 Seeds)') + r'''}\label{tab:lifef1}
\centering\scriptsize\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
 & \multicolumn{4}{c}{Net F1 (coverage) at round} & Mean\\
Scheme & 200 & 400 & 600 & 800 & 1--1000\\
\midrule
Local only & 0.708 (1.00) & 0.685 (0.99) & 0.646 (0.95) & 0.547 (0.85) & 0.631\\
Centralized & 0.699 (0.75) & 0.380 (0.31) & 0 (0) & 0 (0) & 0.284\\
FedAvg-flat & 0.114 (0.08) & 0 (0) & 0 (0) & 0 (0) & 0.080\\
FedAvg-LEACH & 0.259 (0.23) & 0 (0) & 0 (0) & 0 (0) & 0.129\\
FedAvg, matched & 0.826 (1.00) & 0.831 (1.00) & 0.746 (0.82) & 0.027 (0.02) & 0.551\\
EASFL, dense & 0.821 (1.00) & 0.827 (1.00) & 0.666 (0.71) & 0.175 (0.10) & 0.567\\
\textbf{EASFL-M} & 0.827 (1.00) & 0.828 (1.00) & 0.741 (0.82) & 0.027 (0.02) & 0.551\\
\bottomrule
\end{tabular}
\end{table}

\section{Discussion and Limitations}'''
rep('\n\\section{Discussion and Limitations}', NEW)

# ---- discussion / threat model / conclusion ------------------------------------------------------
rep('The real traces carry no expert labels, so faults were injected; validation on traces with expert-labelled faults remains future work.',
    'The real traces carry no expert labels, so faults were injected; ' + R('the Intel Lab experiment is therefore a semi-synthetic validation, and validation on traces with expert-labelled faults remains future work.'))
rep('\\textit{Threat model.} We assume honest CHs and non-colluding, non-adaptive attackers.',
    R('\\textit{Where the gains come from.} Matched baselines (Section~\\ref{sec:matched}) show that the energy and lifetime gains are due to compression, participation and CH selection, and that the mask-aware rule adds robustness under attack and on heterogeneous real data rather than clean accuracy.') + '\n\n\\textit{Threat model.} We assume honest CHs and non-colluding, non-adaptive attackers' + R(', and we claim robustness only up to 20\\% malicious nodes.'))
rep('Future work will validate EASFL on traces with expert-labelled faults,',
    R('Compared with baselines given the same compression and participation, its mask-aware aggregation adds no accuracy without attack but is significantly more robust to sign-flip poisoning on both data sets, and its detection quality holds for the whole period in which nodes survive.') + ' Future work will validate EASFL on traces with expert-labelled faults,')

rep('and with writing the simulation code for those experiments.',
    'and with writing the simulation code for those experiments' + R(', and, in this revision, with the matched-baseline and lifetime experiments (Sections~\\ref{sec:matched} and~\\ref{sec:lifetimef1}), their statistical analysis and the revised text') + '.')

# ======================= AE-1 / AE-2 (compromised CHs, collusion) =======================
rep('CHs are assumed to aggregate honestly; adversarial CHs are left for future work.',
    R('Compromised nodes can also be elected CH. A compromised CH can replace its cluster aggregate (we use $-4\\times$ the honest aggregate) and can report a full battery so that the energy-aware score elects it. Attackers can also collude by sending only a shared set of coordinates. Sections~\\ref{sec:protect} and~\\ref{sec:insider} describe and evaluate the defences; elsewhere CHs are honest.'))

rep('\\subsection{On-Node Personalization}',
    R(r"""\textit{Support-collusion test.} Because the reference of EASFL-M is computed per coordinate from the updates that carry it, attackers that agree to send the same coordinates could dominate those coordinates in a small cluster. Honest top-$k$ supports overlap little: over 5{,}047 honest pairs in one cluster run their Jaccard similarity had median 0.12 and maximum 0.56, while colluders sending a shared set have similarity 1. The CH therefore also flags any update whose support has Jaccard similarity above $\tau=0.7$ with another update in the same round. The test needs no communication and costs $O(|U|^2K)$ operations at the CH.""") + r"""

\subsection{""" + R('Protecting the Aggregation Path') + r"""}\label{sec:protect}
""" + R(r"""The CH score of Section~\ref{sec:method} uses residual energy, which nodes report themselves, and the CH aggregates on behalf of its members. Both can be abused by a compromised node. The BS keeps every node's suspicion score, strike count and quarantine state and broadcasts the quarantine list together with the model (a few bytes per round, not counted in the traffic figures). It applies three checks. (i) \emph{Member audit}: in each cluster one participant, chosen by the BS at random after the round, also sends its compressed update (746 bits) directly to the BS; the CH aggregate is rejected if its cosine with that update on the update's support is negative, because an honest aggregate contains that update. Each rejection is a strike against the CH, and two strikes quarantine it for 100 rounds. (ii) \emph{Energy bound}: the BS knows each node's roles and so a lower bound on the energy it has spent; a reported residual energy above the largest initial energy minus that bound is capped at the bound. (iii) Optionally, a \emph{CH cooldown} that lets a node serve as CH at most once every $1/p$ rounds, and BS-level screening of cluster aggregates by norm and direction. All checks reuse quantities the protocol already has.""") + r"""

\subsection{On-Node Personalization}""")

INSIDER = r"""
\subsection{""" + R('Coordinate-Targeting Collusion and Compromised Cluster Heads') + r"""}\label{sec:insider}
""" + R(r"""Table~\ref{tab:insider} evaluates the two insider threats over 10 seeds. Every rule uses the same compression and participation as in Section~\ref{sec:matched}.

\textit{Coordinate-targeting collusion.} Attackers send only a shared set of 42 coordinates with shared signs, scaled to 2.5 times the norm of their own honest update so that they pass the norm test. This attack lowers EASFL-M from 0.667 (clean, same seeds) to 0.488 at 20\% on the synthetic workload, as the editor anticipated. The support-collusion test restores 0.590 at 20\% and 0.575 at 30\%, significantly above every other rule (paired Wilcoxon, Holm $p\le0.012$ at 20\% and 30\%; on Intel Lab 0.536 against at most 0.458, Holm $p=0.0098$). Its clean cost is nil (0.667 and 0.574). An adaptive colluder that mixes a random half of the shared set with its own top coordinates keeps its support similarity at honest levels and evades the test. Its push is diluted, so EASFL-M still scores 0.596 at 20\% on the synthetic workload, but here the dense filter (0.649) and matched trimmed mean (0.624) do better; on Intel Lab all rules lie within 0.03 of each other. We report this as a limitation.

\textit{Compromised cluster heads.} Without a defence, compromised CHs are decisive: at 20\% malicious nodes EASFL-M falls to 0.401 when CHs flip their aggregates, and to 0.09 on the synthetic workload and 0.15 on Intel Lab when they also report a full battery, because the energy-aware score then elects them in most rounds (80\% of CH slots). Matched FedAvg collapses in the same way. The energy bound alone does not help, because an adaptive liar reports exactly the bound, and BS screening alone does not help when a malicious CH sits in every cluster round. The member audit restores 0.683 (synthetic) and 0.570 (Intel Lab) under the strongest attack, close to the clean values (0.667 and 0.574), at an energy cost of 4.7\% and 1.5\% over 200 rounds and an FND of 525 instead of 546. With all checks together the result holds up to 30\% malicious nodes (0.671). The cooldown alone recovers much of the loss on the synthetic workload but not on Intel Lab, and it shortens FND from 546 to 324 because it forces weaker nodes into the CH role, so we recommend the audit and the energy bound, with the cooldown as an option.""") + r"""

\begin{table*}[t]
\caption{""" + R('Global F1 Under Insider Attacks (Mean $\\pm$ Std Over 10 Seeds; Same Compression and Participation for All Rules). Clean EASFL-M on the Same Seeds: 0.667 (Synthetic), 0.574 (Intel Lab)') + r"""}\label{tab:insider}
\centering\scriptsize\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\multicolumn{7}{@{}l}{(a) Coordinate-targeting collusion (shared support), and adaptive colluders (half shared support)}\\
 & \multicolumn{3}{c}{Synthetic, shared support} & Intel, shared & Synthetic, adaptive & Intel, adaptive\\
Rule & 10\% & 20\% & 30\% & 20\% & 20\% & 20\%\\
\midrule
Matched FedAvg & \pms{0.550}{0.067} & \pms{0.396}{0.037} & \pms{0.356}{0.051} & \pms{0.385}{0.060} & \pms{0.558}{0.048} & \pms{0.509}{0.048}\\
Matched trimmed mean & \pms{0.593}{0.091} & \pms{0.424}{0.099} & \pms{0.319}{0.050} & \pms{0.382}{0.069} & \pms{0.624}{0.061} & \pms{0.503}{0.050}\\
Matched support-aware median & \pms{0.570}{0.080} & \pms{0.462}{0.083} & \pms{0.399}{0.066} & \pms{0.351}{0.073} & \pms{0.579}{0.059} & \pms{0.494}{0.063}\\
EASFL, dense filter & \pms{0.577}{0.099} & \pms{0.430}{0.094} & \pms{0.345}{0.093} & \pms{0.369}{0.156} & \pms{\textbf{0.649}}{0.058} & \pms{0.505}{0.032}\\
EASFL-M & \pms{0.576}{0.067} & \pms{0.488}{0.065} & \pms{0.427}{0.047} & \pms{0.458}{0.059} & \pms{0.596}{0.049} & \pms{\textbf{0.520}}{0.044}\\
\textbf{EASFL-M + support test} & \pms{\textbf{0.625}}{0.064} & \pms{\textbf{0.590}}{0.051} & \pms{\textbf{0.575}}{0.095} & \pms{\textbf{0.536}}{0.041} & \pms{0.596}{0.049} & \pms{0.517}{0.044}\\
\midrule
\multicolumn{7}{@{}l}{(b) Compromised CHs: flipped aggregate (CH), plus false full-battery reports (F), plus sign-flipping members (M)}\\
 & \multicolumn{3}{c}{Synthetic, CH+F} & Synthetic, CH+F+M & Intel, CH+F & Intel, CH+F+M\\
Defence & 10\% & 20\% & 30\% & 20\% & 20\% & 20\%\\
\midrule
None (EASFL-M) & \pms{0.088}{0.064} & \pms{0.093}{0.063} & \pms{0.082}{0.062} & \pms{0.116}{0.056} & \pms{0.147}{0.052} & \pms{0.156}{0.029}\\
Matched FedAvg, no defence & -- & \pms{0.076}{0.064} & -- & \pms{0.098}{0.065} & \pms{0.147}{0.052} & \pms{0.130}{0.068}\\
Energy bound & -- & \pms{0.095}{0.061} & -- & -- & \pms{0.147}{0.052} & --\\
BS screening & -- & \pms{0.082}{0.054} & -- & -- & \pms{0.148}{0.043} & --\\
Cooldown & -- & \pms{0.579}{0.093} & -- & -- & \pms{0.176}{0.036} & --\\
Screening + cooldown + bound & \pms{0.671}{0.061} & \pms{0.655}{0.073} & \pms{0.252}{0.107} & \pms{0.655}{0.072} & \pms{0.267}{0.099} & \pms{0.518}{0.046}\\
\textbf{Member audit} & -- & \pms{\textbf{0.683}}{0.073} & -- & -- & \pms{\textbf{0.570}}{0.038} & --\\
All checks + audit & \pms{0.663}{0.071} & \pms{0.680}{0.057} & \pms{\textbf{0.671}}{0.078} & \pms{\textbf{0.666}}{0.069} & \pms{0.565}{0.036} & \pms{\textbf{0.558}}{0.019}\\
\bottomrule
\end{tabular}
\end{table*}
"""
rep('\n\\subsection{Detection Over the Whole Network Lifetime}\\label{sec:lifetimef1}', INSIDER + '\n\\subsection{Detection Over the Whole Network Lifetime}\\label{sec:lifetimef1}')

rep('\\textit{Threat model.} We assume honest CHs and non-colluding, non-adaptive attackers' + R(', and we claim robustness only up to 20\\% malicious nodes.') + ' Adaptive attacks crafted against the specific defence~\\cite{fang2020}, backdoors~\\cite{bagdasaryan2020} and compromised CHs remain open.',
    '\\textit{Threat model.} ' + R('Compromised CHs and coordinate-targeting collusion are handled by the member audit and the support-collusion test (Section~\\ref{sec:insider}). The audit assumes that a CH cannot predict which member will be audited; an adaptive colluder that randomizes its support evades the support test, and other attacks crafted against the specific defence~\\cite{fang2020} and backdoors~\\cite{bagdasaryan2020} remain open. Without these checks we claim robustness only up to 20\\% malicious nodes.'))

rep('and its detection quality holds for the whole period in which nodes survive.',
    'and its detection quality holds for the whole period in which nodes survive. ' + R('A member audit and a support-collusion test extend this protection to compromised cluster heads that lie about their energy and to colluding attackers, at about 2--5\\% extra energy.'))


# ======================= stage 3: 20-seed tables, breakdown, generalization, model size =======================
rep('\\newcommand{\\pms}[2]{#1{\\tiny$\\pm$#2}}',
    '\\newcommand{\\pms}[2]{#1{\\tiny$\\pm$#2}}\n\\newcommand{\\revtab}{\\ifdefined\\cleanversion\\else\\color{blue}\\fi}')

# --- abstract numbers (20 seeds)
rep('delays the first node death from round 35 to 547, and attains a network-level F1-score of 0.82 at round 200 against 0.26,',
    'delays the first node death from round ' + R('34 to 558') + ', and attains a network-level F1-score of ' + R('0.83') + ' at round 200 against ' + R('0.27') + ',')
rep('uses $16\\times$ less traffic and $3.3\\times$ less energy, matches the per-node F1 of hierarchical FedAvg (0.56), and raises network-level F1 from 0.32 to 0.57.',
    'uses $16\\times$ less traffic and ' + R('$3.2\\times$') + ' less energy, matches the per-node F1 of hierarchical FedAvg (' + R('0.57') + '), and raises network-level F1 from ' + R('0.34') + ' to 0.57.')

# --- Table II (20 seeds)
rep('\\caption{Detection and Efficiency After 200 Rounds (Mean $\\pm$ Std Over 5 Seeds). G: Global Model; P: Personalized; Net: Network-Level F1 With Dead Nodes Silent}\\label{tab:main}\n\\centering',
    '\\caption{Detection and Efficiency After 200 Rounds (Mean $\\pm$ Std Over ' + R('20') + ' Seeds). G: Global Model; P: Personalized; Net: Network-Level F1 With Dead Nodes Silent}\\label{tab:main}\n\\centering\\revtab')
i0 = src.index('Centralized & \\pmv{0.783}{0.021}'); i1 = src.index('\\bottomrule', i0)
src = src[:i0] + r"""Centralized & \pmv{0.775}{0.023} & 0.940 & 0.023 & \pmv{0.807}{0.018} & 0.015 & \pmv{0.695}{0.024} & 136.9 & 26.3 & 74.8 & 28\\
Local only & \pmv{0.688}{0.023} & 0.881 & 0.030 & \pmv{0.703}{0.026} & 0.019 & \pmv{0.703}{0.026} & 0 & -- & 100 & --\\
FedAvg-flat & \pmv{0.591}{0.089} & 0.917 & 0.094 & \pmv{0.815}{0.019} & 0.015 & \pmv{0.120}{0.038} & 122.3 & 49.5 & 8.1 & 18\\
FedAvg-LEACH & \pmv{0.660}{0.062} & 0.927 & 0.059 & \pmv{0.825}{0.015} & 0.013 & \pmv{0.274}{0.041} & 197.4 & 47.8 & 21.6 & 34\\
\textbf{EASFL-M} & \pmv{0.680}{0.059} & 0.927 & 0.051 & \pmv{0.825}{0.016} & 0.013 & \pmv{\textbf{0.825}}{0.016} & \textbf{15.3} & \textbf{15.4} & \textbf{100} & \textbf{558}\\
""" + src[i1:]
rep('Without personalization, the hierarchical FL schemes reach a per-node F1 of about 0.71, below centralized learning (0.78) but above purely local models (0.69). FedAvg-flat is worse (0.59) because its participants die quickly and the model degrades after round 120 (Fig.~\\ref{fig:conv}). EASFL matches FedAvg-LEACH in F1 (0.711 vs.\\ 0.706) with a lower FPR (0.030 vs.\\ 0.038) and lower variance across seeds, while transmitting 15.2\\,Mbit instead of 197.1\\,Mbit ($13\\times$ less) and consuming 15.6\\,J instead of 47.3\\,J ($3\\times$ less) over 200 rounds.',
    R('With 20 seeds, without personalization, the hierarchical FL schemes reach a per-node F1 of 0.66--0.68, below centralized learning (0.78) and close to purely local models (0.69). FedAvg-flat is worse (0.59) because its participants die quickly and the model degrades after round 120 (Fig.~\\ref{fig:conv}). EASFL-M is slightly above FedAvg-LEACH in F1 (0.680 vs.\\ 0.660) with a lower FPR (0.051 vs.\\ 0.059), while transmitting 15.3\\,Mbit instead of 197.4\\,Mbit ($13\\times$ less) and consuming 15.4\\,J instead of 47.8\\,J ($3\\times$ less) over 200 rounds. After personalization all FL schemes reach 0.82--0.83.'))
rep('By round 200 only 7.6 FedAvg-flat nodes and 22.6 FedAvg-LEACH nodes are alive, so most anomalies go unreported and their network-level F1 falls to 0.12 and 0.26. Centralized collection retains 75 nodes (network F1 0.70). EASFL keeps all 100 nodes alive and its network-level F1 equals its per-node F1, 0.82.',
    R('By round 200 only 8.1 FedAvg-flat nodes and 21.6 FedAvg-LEACH nodes are alive, so most anomalies go unreported and their network-level F1 falls to 0.12 and 0.27. Centralized collection retains 75 nodes (network F1 0.70). EASFL-M keeps all 100 nodes alive and its network-level F1 equals its per-node F1, 0.83.'))

# --- Table III (20 seeds)
rep('\\caption{Lifetime (Rounds) and Energy in the First 200 Rounds}\\label{tab:life}\n\\centering',
    '\\caption{Lifetime (Rounds) and Energy in the First 200 Rounds ' + R('(Mean of 20 Seeds)') + '}\\label{tab:life}\n\\centering\\revtab')
i0 = src.index('Centralized (LEACH) & 25 & 320 & 489 & 25.9\\\\'); i1 = src.index('\\bottomrule', i0)
src = src[:i0] + r"""Centralized (LEACH) & 28 & 320 & 487 & 26.3\\
Centralized (EA-CH) & 146 & 252 & 462 & 39.3\\
FedAvg-flat & 18 & 74 & 186 & 49.5\\
FedAvg-LEACH & 34 & 148 & 235 & 47.8\\
\textbf{EASFL} & \textbf{558} & 660 & 742 & \textbf{15.4}\\
\quad w/o participant selection & 317 & 548 & 628 & 19.2\\
\quad w/o compression & 223 & 252 & 294 & 40.5\\
\quad w/o energy-aware CH (LEACH) & 160 & \textbf{702} & \textbf{803} & 15.9\\
""" + src[i1:]
rep('EASFL delays the first node death to round 547, against 35 for FedAvg-LEACH (15.6$\\times$) and 25 for centralized collection, and its HND (645) is 4.4$\\times$ that of FedAvg-LEACH.',
    R('EASFL delays the first node death to round 558, against 34 for FedAvg-LEACH (16$\\times$) and 28 for centralized collection, and its HND (660) is 4.5$\\times$ that of FedAvg-LEACH (20 seeds; the lifetime curves are identical for EASFL and EASFL-M because aggregation has no radio cost).'))
rep('Compression is the largest contributor: without it, energy use rises 2.6$\\times$ and HND falls from 645 to 248. Participant selection adds 73\\% to FND. Energy-aware CH selection triples FND (547 vs.\\ 177) but slightly shortens HND and LND relative to LEACH rotation (645 vs.\\ 697, 731 vs.\\ 802).',
    R('Compression is the largest contributor: without it, energy use rises 2.6$\\times$ and HND falls from 660 to 252. Participant selection adds 76\\% to FND. Energy-aware CH selection raises FND 3.5-fold (558 vs.\\ 160) but slightly shortens HND and LND relative to LEACH rotation (660 vs.\\ 702, 742 vs.\\ 803).'))
rep('(FND 25 $\\rightarrow$ 142, HND 320 $\\rightarrow$ 246).', R('(FND 28 $\\rightarrow$ 146, HND 320 $\\rightarrow$ 252).'))

# --- Table IV (20 seeds)
rep('\\caption{F1 With 20\\% Malicious Nodes: Global Model / Personalized}\\label{tab:pois}\n\\centering',
    '\\caption{F1 With 20\\% Malicious Nodes: Global Model / Personalized ' + R('(Mean of 20 Seeds; Baselines at Full Precision)') + '}\\label{tab:pois}\n\\centering\\revtab')
i0 = src.index('FedAvg (mean) & 0.24 / 0.40'); i1 = src.index('\\bottomrule', i0)
src = src[:i0] + r"""FedAvg (mean) & 0.17 / 0.30 & 0.62 / 0.80 & 0.40 / 0.82 & 0.66 / 0.82\\
Median & 0.60 / 0.81 & 0.60 / 0.81 & 0.52 / 0.82 & 0.60 / 0.82\\
Trimmed mean & 0.60 / 0.79 & 0.62 / 0.81 & 0.55 / 0.82 & 0.63 / 0.82\\
Krum & 0.45 / 0.75 & 0.36 / 0.72 & 0.45 / 0.75 & 0.43 / 0.74\\
EASFL-M w/o robust & 0.21 / 0.39 & 0.65 / 0.81 & 0.40 / 0.82 & 0.68 / 0.82\\
EASFL-M w/o cosine & 0.65 / 0.81 & 0.65 / 0.81 & 0.61 / 0.83 & 0.68 / 0.82\\
\textbf{EASFL-M} & \textbf{0.64} / 0.79 & \textbf{0.66} / 0.81 & \textbf{0.61} / 0.82 & \textbf{0.68} / 0.82\\
""" + src[i1:]
rep('Undefended FedAvg collapses under the scaled sign flip (F1 0.24 at 20\\%, 0.13 at 30\\%). With 10\\% and 20\\% attackers, EASFL keeps global F1 between 0.64 and 0.71 across all attacks, within 0.07 of its clean value (0.71) and higher than every compared aggregation rule in all eight cases. Removing EASFL\'s filtering and quarantine (``EASFL w/o robust\'\') lets the sign-flip attack succeed (0.18), confirming that the defence and not compression or participant selection provides the resilience. The label-flip attack is notable: its updates are neither large nor reversed in direction, yet EASFL (0.67) outperforms median (0.53) and trimmed mean (0.57), because repeated flags quarantine persistent offenders. Krum performs poorly throughout (0.35--0.52), even without attackers,',
    R('Undefended FedAvg collapses under the scaled sign flip (F1 0.17 at 20\\%, 20 seeds). With 20\\% attackers EASFL-M keeps global F1 between 0.61 and 0.68 across the four attacks, within 0.07 of its clean value (0.68) and higher than every full-precision rule in all four cases. Removing the filtering and quarantine (``EASFL-M w/o robust\'\') lets the sign-flip attack succeed (0.21), confirming that the defence and not compression or participant selection provides the resilience. The label-flip attack is notable: its updates are neither large nor reversed in direction, yet EASFL-M (0.61) outperforms median (0.52) and trimmed mean (0.55), because repeated flags quarantine persistent offenders. Krum performs poorly throughout (0.36--0.45), even without attackers,'))
rep('At 30\\% attackers EASFL becomes unstable across seeds (sign flip: F1 0.50 $\\pm$ 0.25; stealth: 0.46 $\\pm$ 0.21), whereas the coordinate-wise median degrades more gracefully (0.55 in both cases).',
    R('At 30\\% attackers EASFL-M becomes unstable across seeds (sign flip: F1 0.46 $\\pm$ 0.25; stealth: 0.51 $\\pm$ 0.16), whereas the full-precision coordinate-wise median degrades more gracefully (0.55 in both cases).'))
rep('The cosine test added little beyond the norm test and quarantine (Table~\\ref{tab:pois}); it is cheap and retained, but its effect is not statistically distinguishable in our runs.',
    R('The direction (cosine) test adds nothing measurable beyond the norm test and quarantine: with 20 seeds, removing it changes F1 by at most 0.01 under the four attacks (Table~\\ref{tab:pois}), and also under an inner-product-manipulation attack ($-0.5\\Delta_i$, small and opposite) and the stealth attack, for which the direction test is designed (10 seeds, 20\\% and 30\\%; differences $\\le0.009$, Wilcoxon $p\\ge0.23$). The norm test catches scaled updates, the quarantine catches persistent offenders, and a small opposite update is diluted by the trimmed mean. We keep the test because it costs one dot product per update and all results include it, but we do not claim a benefit from it.'))

# --- Table V (Intel, 20 seeds)
rep('\\caption{Intel Lab Deployment (50 Motes, Real Readings and Layout, Injected Faults; Mean of 5 Seeds). ' + R('--: Not Applicable or Not Measured in This Run') + '}\\label{tab:intel}\n\\centering',
    '\\caption{Intel Lab Deployment (50 Motes, Real Readings and Layout, Injected Faults; Mean of ' + R('20') + ' Seeds). ' + R('Local Only: One-Time Training, No Radio Use') + '}\\label{tab:intel}\n\\centering\\revtab')
i0 = src.index('Centralized (LEACH) & $0.543\\pm0.021$'); i1 = src.index('\\bottomrule', i0)
src = src[:i0] + r"""Centralized (LEACH) & $0.566\pm0.031$ & $0.571\pm0.029$ & $0.507\pm0.034$ & 70.7 & 8.1 & 40.9 & 43 & 515\\
Local only & $0.407\pm0.045$ & $0.391\pm0.045$ & $0.391\pm0.045$ & 0 & 0.6 & 50.0 & $>$4000 & $>$4000\\
FedAvg-flat & $0.569\pm0.028$ & $0.566\pm0.033$ & $0.285\pm0.083$ & 109.0 & 23.1 & 16.1 & 79 & 168\\
FedAvg-LEACH & $0.571\pm0.031$ & $0.571\pm0.031$ & $0.340\pm0.057$ & 113.4 & 20.9 & 24.2 & 38 & 194\\
EASFL (dense filter) & $0.532\pm0.030$ & $0.535\pm0.032$ & $0.535\pm0.032$ & \textbf{7.1} & \textbf{6.4} & \textbf{50.0} & \textbf{609} & \textbf{800}\\
\textbf{EASFL-M} (mask-aware) & $\mathbf{0.570\pm0.032}$ & $\mathbf{0.567\pm0.033}$ & $\mathbf{0.567\pm0.033}$ & 7.2 & \textbf{6.4} & \textbf{50.0} & \textbf{609} & \textbf{800}\\
\quad w/o compression & $0.572\pm0.029$ & $0.569\pm0.034$ & $0.568\pm0.035$ & 71.8 & 16.8 & 49.4 & 230 & 312\\
\quad w/o robust filtering & $0.569\pm0.035$ & $0.562\pm0.033$ & $0.562\pm0.033$ & 7.2 & 6.4 & 50.0 & 609 & 800\\
\quad w/o participant selection & $0.583\pm0.030$ & $0.570\pm0.034$ & $0.570\pm0.034$ & 11.6 & 8.3 & 50.0 & 345 & 628\\
""" + src[i1:]
rep('EASFL transmits $16\\times$ less traffic and uses $3.3\\times$ less energy than FedAvg-LEACH over 200 rounds, keeps all 50 motes alive, and delays the first node death from round 40 to 619 and the half-network death from 197 to 803. Its network-level F1 (0.53) is the highest of all schemes, against 0.32 for FedAvg-LEACH and 0.52 for centralized collection, which has lost 9 motes by round 200.',
    R('With 20 seeds, EASFL-M transmits $16\\times$ less traffic and uses $3.2\\times$ less energy than FedAvg-LEACH over 200 rounds, keeps all 50 motes alive, and delays the first node death from round 38 to 609 and the half-network death from 194 to 800. Its network-level F1 (0.57) is the highest of all schemes, against 0.34 for FedAvg-LEACH and 0.51 for centralized collection, which has lost 9 motes by round 200. Local-only models never transmit and so never die from radio use, but detect poorly on real data (0.39).'))
rep('First, EASFL with the dense filter has a per-node F1 0.04 below FedAvg-LEACH (0.519 vs.\\ 0.556). The ablation locates the cause: removing either compression or robust filtering restores it (0.560 and 0.557).',
    R('First, EASFL with the dense filter has a per-node F1 0.04 below FedAvg-LEACH (0.532 vs.\\ 0.571, 20 seeds). The ablation locates the cause: removing either compression or the dense filter restores it.'))
rep('EASFL-M removes this loss (Section~\\ref{sec:mask}): 0.561 per-node and 0.567 network-level F1, with no honest node quarantined. Second, on-node personalization gives little on real data (0.519 to 0.530, and no gain for the other schemes),',
    R('EASFL-M removes this loss (Section~\\ref{sec:mask}): 0.570 per-node and 0.567 network-level F1, equal to FedAvg-LEACH, with no honest node quarantined. Second, on-node personalization gives little on real data (no gain for any scheme),'))

# --- new subsection: breakdown, generalization, model size, sensitivity (before the insider subsection)
EXTRA = r"""
\subsection{""" + R('Energy Breakdown, Unseen Faults, Detector Choice and Sensitivity') + r"""}\label{sec:extra}
""" + R(r"""\textit{Where the energy goes.} Table~\ref{tab:breakdown} splits the energy of the first 200 rounds by protocol component. Every scheme is charged for receiving as well as transmitting: CH advertisements, join messages and the model broadcast are received by every alive node. For EASFL the broadcast of the 16-bit model to all nodes is the largest item (44\%), followed by training (20\%), CH-to-BS uplink (16\%) and control messages for re-clustering every round (14\%); member uploads are only 3\%. Matched FedAvg has the same breakdown, consistent with Section~\ref{sec:matched}. The member audit adds 0.7\,J (4.7\%). Re-clustering less often and broadcasting the model less often are therefore the next levers, not further compression of the uploads.

\textit{Unseen fault types.} To test transfer across fault mechanisms, we removed one fault family from every node's training windows and kept all four in the test windows (synthetic, 5 seeds; Supplementary Table~S2). Recall on the held-out family falls from 0.58 to 0.35 for spikes, 0.89 to 0.51 for drift and 0.72 to 0.60 for noise bursts, and stays at 1.00 for stuck-at faults. The features therefore transfer only partly to an unseen mechanism, and in deployment the detector should be retrained when a new fault type appears. Centralized learning shows the same pattern, so this is a property of the features and labels, not of federation.

\textit{Detector choice and model size.} On the pooled synthetic training data (centralized, 3 seeds; Supplementary Table~S3), the 417-parameter MLP on summary features (F1 0.78) outperforms an isolation forest on the same features (0.66), a GRU classifier with 1{,}025 parameters (0.63), a 1D-CNN classifier with 1{,}025 parameters (0.51) and a 1D-CNN autoencoder (0.36) on the raw 10-sample windows. Larger federated MLPs keep the robustness of EASFL-M (2{,}081 parameters: 0.48 against 0.18 for matched FedAvg under 20\% sign flip) but multiply radio energy: with 2{,}081 parameters only 26 of 100 nodes are alive at round 200, and with 10{,}401 none are. Model size, not only compression, must therefore be chosen against the energy budget.

\textit{Thresholds and personalization energy.} Choosing each node's threshold on a held-out 20\% validation split instead of its training windows lowers per-node F1 from 0.830 to 0.769, because fine-tuning and threshold selection then use fewer windows; the reported test F1 is computed on unseen test windows in both cases, so the training-window threshold does not inflate it. Personalization costs about 600 sample-passes (1.2\,mJ) per node once, 0.12\,J for the whole network, which is included in no table and is below 1\% of $E_{200}$.

\textit{Sensitivity.} Supplementary Table~S1 varies node count (50, 200), field size (200\,m), BS position (inside the field, 300\,m away), CH fraction (0.05, 0.2), radio electronics energy (25, 100\,nJ/bit) and training energy (0.5, 8\,$\mu$J). EASFL has the longest FND in every setting. Its advantage over centralized collection is largest when the BS is far and smallest when the BS is inside the field, where centralized collection reaches a longer LND (1{,}178 vs.\ 823 rounds) although EASFL keeps a longer FND (561 vs.\ 43) and HND (761 vs.\ 682). Higher training energy (8\,$\mu$J per sample-pass) shortens EASFL's FND from 547 to 415 rounds, and a denser network (200 nodes) from 547 to 210, because the per-round control and broadcast reception grows with the number of nodes.""") + r"""

\begin{table}[t]
\caption{""" + R('Energy by Protocol Component Over 200 Rounds (J, Synthetic, Mean of 5 Seeds)') + r"""}\label{tab:breakdown}
\centering\scriptsize\setlength{\tabcolsep}{2.2pt}\revtab
\begin{tabular}{@{}lccccccc@{}}
\toprule
Scheme & Train & Uplink & CH rx & CH$\to$BS & Control & Broadcast & Total\\
\midrule
Centralized (raw) & -- & 5.49 & 3.23 & 17.32 & 0.93 & -- & 26.97\\
FedAvg-flat & 2.69 & 40.64 & -- & -- & -- & 6.08 & 49.41\\
FedAvg-LEACH & 4.37 & 14.09 & 9.20 & 9.75 & 0.81 & 9.86 & 48.06\\
EASFL-M & 3.15 & 0.43 & 0.43 & 2.51 & 2.19 & 6.87 & 15.59\\
\bottomrule
\end{tabular}
\end{table}
"""
rep('\n\\subsection{' + R('Coordinate-Targeting Collusion and Compromised Cluster Heads') + '}\\label{sec:insider}',
    EXTRA + '\n\\subsection{' + R('Coordinate-Targeting Collusion and Compromised Cluster Heads') + '}\\label{sec:insider}')

# --- conclusion numbers
rep('Measured at the network level, where dead nodes cannot report anomalies, EASFL detects anomalies with F1 0.82 at round 200 while hierarchical FedAvg reaches 0.26.',
    'Measured at the network level, where dead nodes cannot report anomalies, EASFL detects anomalies with F1 ' + R('0.83') + ' at round 200 while hierarchical FedAvg reaches ' + R('0.27') + '.')
rep('reaches a per-node F1 of 0.56, on par with hierarchical FedAvg, while raising network-level F1 from 0.32 to 0.57.',
    'reaches a per-node F1 of ' + R('0.57') + ', on par with hierarchical FedAvg, while raising network-level F1 from ' + R('0.34') + ' to 0.57.')


# ======================= stage 4: FIT IoT-LAB measurement (AE-4, R1-4, R2-2) =======================
TESTBED = r"""
\subsection{""" + R('Measurement on IEEE 802.15.4 Hardware') + r"""}\label{sec:testbed}
""" + R(r"""To check that the first-order radio model does not decide the ranking, we replayed federated rounds on 20 FIT IoT-LAB M3 nodes at the Grenoble site~\cite{adjih2015} (ARM Cortex-M3, AT86RF231 IEEE 802.15.4 radio, CSMA/CA with acknowledgements and up to three retransmissions) while the testbed recorded each node's power. One node acted as BS, one as CH and 18 as members of one cluster. Each round consisted of a CH advertisement, a join from every member, the member uploads, the CH-to-BS upload and the model broadcast, with payloads equal to the bit counts of Table~\ref{tab:params} sent as frames of up to 100 bytes (EASFL: 50\% of members send 94\,B, one frame; FedAvg-LEACH: all members send 1{,}668\,B, 17 frames; centralized: all members send 480\,B of raw samples, 5 frames). Members transmitted at the same time, so contention, collisions and retransmissions occurred as they would in the field. We ran 100 rounds of each scheme (experiment 459032).

Table~\ref{tab:testbed} reports the energy per round with the radio switched on only while a node sends or receives: the measured energy above the listening level plus the node's measured listening power (0.122\,W for the whole board) times the time its radio was busy. Measured on hardware, FedAvg-LEACH consumes $6.0\times$ and centralized collection $2.2\times$ the energy of EASFL per round, larger gaps than the first-order model predicts ($3.1\times$ and $1.7\times$). Contention explains the difference. Large uploads keep the channel busy, so FedAvg members delivered only 18\% of their frames to the CH after retries, against 38\% for EASFL, and a 17-frame FedAvg update is far less likely than a one-frame EASFL update to arrive complete. Losses are high for every scheme because 18 members contend for one CH along a corridor, which is a demanding but fair setting. Re-running the lifetime simulation with the measured radio time per frame (30\,ms, including backoff and retries) and the AT86RF231 radio currents keeps EASFL first in FND and HND (18 and 33 rounds against at most 7 and 29 for the other schemes); absolute lifetimes become short because the initial energies of Table~\ref{tab:params} were sized for the first-order model, and centralized collection then reaches a longer LND (51 vs.\ 37 rounds). The measurement covers one cluster at one site; multi-cluster interference and duty-cycled MAC protocols remain to be measured.""") + r"""

\begin{table}[t]
\caption{""" + R('Energy per Round Measured on 20 FIT IoT-LAB M3 Nodes (100 Rounds per Scheme; Radio On Only While a Node Sends or Receives)') + r"""}\label{tab:testbed}
\centering\scriptsize\setlength{\tabcolsep}{2.2pt}\revtab
\begin{tabular}{@{}lccc@{}}
\toprule
 & EASFL & FedAvg-LEACH & Centralized\\
\midrule
Upload per member (bytes / frames) & 94 / 1 & 1{,}668 / 17 & 480 / 5\\
CH energy per round (mJ) & 22.9 & 102.1 & 102.3\\
Participating member (mJ) & 18.3 & 91.3 & 29.3\\
Non-participating member (mJ) & 11.4 & -- & --\\
CH radio busy time per round (s) & 0.19 & 0.83 & 0.76\\
Member frames delivered after retries & 38\% & 18\% & 20\%\\
\textbf{Cluster energy per round (mJ)} & \textbf{290} & 1{,}745 & 630\\
Ratio to EASFL, measured & 1.0 & 6.0 & 2.2\\
Ratio to EASFL, first-order model & 1.0 & 3.1 & 1.7\\
\bottomrule
\end{tabular}
\end{table}
"""
rep('\n\\subsection{' + R('Coordinate-Targeting Collusion and Compromised Cluster Heads') + '}\\label{sec:insider}',
    TESTBED + '\n\\subsection{' + R('Coordinate-Targeting Collusion and Compromised Cluster Heads') + '}\\label{sec:insider}')
rep('\\textit{Simulator scope.} The round-based simulator abstracts MAC contention, retransmissions, idle listening and sleep scheduling, which affect all schemes but may change absolute lifetimes.',
    '\\textit{Simulator scope.} The round-based simulator abstracts MAC contention, retransmissions, idle listening and sleep scheduling. ' + R('The hardware measurement of Section~\\ref{sec:testbed} shows that these effects change absolute lifetimes but not the ranking, and that contention widens the gap in favour of EASFL.'))
rep('and two-tier robust aggregation with quarantine maintains accuracy with up to 20\\% malicious nodes on the synthetic workload.',
    'and two-tier robust aggregation with quarantine maintains accuracy with up to 20\\% malicious nodes on the synthetic workload. ' + R('On 20 IEEE 802.15.4 nodes of the FIT IoT-LAB testbed, hierarchical FedAvg used $6.0\\times$ the energy of EASFL per round, a larger gap than the radio model predicts.'))
rep('\\section*{Acknowledgment}\n',
    '\\section*{Acknowledgment}\n' + R('The hardware measurements were carried out on the FIT IoT-LAB testbed~\\cite{adjih2015}. ') )
rep('their statistical analysis and the revised text}.',
    'their statistical analysis and the revised text, and with the testbed firmware and analysis scripts}.')


# ======================= stage 5: novelty table (R1-2, AE-7), trust/IDS (R1-3), CH-policy rule (R2-6c) =======================
NOVELTY = r"""

\begin{table}[t]
\caption{""" + R('Comparison With Related Work') + r"""}\label{tab:novelty}
\centering\scriptsize\setlength{\tabcolsep}{2.4pt}\revtab
\begin{tabular}{@{}lccccccc@{}}
\toprule
Work & E-sel. & Compr. & Robust & Sparse & Quar. & Pers. & Life\\
\midrule
Krum, median, TM~\cite{blanchard2017,yin2018} & -- & -- & \checkmark & -- & -- & -- & --\\
FLTrust~\cite{cao2021} & -- & -- & \checkmark & -- & -- & -- & --\\
FLOW~\cite{liu2024flow} & -- & -- & \checkmark & -- & $\circ$ & -- & --\\
Hu \textit{et al.}~\cite{hu2024} & -- & \checkmark & \checkmark & $\circ$ & -- & -- & --\\
Reputation consensus~\cite{rashid2025} & -- & -- & \checkmark & -- & \checkmark & -- & --\\
Robust personalized FL~\cite{cao2024,thein2024,sun2025} & -- & -- & \checkmark & -- & -- & \checkmark & --\\
Energy-aware selection~\cite{trindade2024} & \checkmark & -- & -- & -- & -- & -- & --\\
Adaptive compression~\cite{marnissi2024} & $\circ$ & \checkmark & -- & -- & -- & -- & --\\
Clustered FL routing~\cite{baqer2025} & $\circ$ & -- & -- & -- & -- & -- & --\\
FLAR~\cite{hawkinson2025} & \checkmark & \checkmark & -- & -- & -- & -- & $\circ$\\
Bhagyalakshmi \textit{et al.}~\cite{bhagyalakshmi2026} & \checkmark & \checkmark & $\circ$ & -- & -- & -- & --\\
TEA-FL~\cite{reis2026} & \checkmark & -- & $\circ$ & -- & $\circ$ & -- & --\\
\textbf{EASFL-M (this work)} & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark\\
\bottomrule
\end{tabular}

\vspace{1mm}\parbox{\columnwidth}{\scriptsize E-sel.: energy-aware participant or head selection; Compr.: update compression; Robust: Byzantine-robust aggregation; Sparse: robust aggregation that treats unsent coordinates as missing rather than zero; Quar.: reputation or quarantine of nodes; Pers.: personalization; Life: node deaths (FND/HND/LND) reported under a radio energy model. \checkmark: addressed; $\circ$: partly (e.g.\ trust-weighted averaging, cosine tests on compressed updates, lifetime as a design goal without node-death results); --: not addressed, as described in the cited works.}
\end{table}
"""
rep('Cryptographic secure aggregation~\\cite{bonawitz2017} protects update confidentiality but not integrity and is orthogonal to our design.',
    'Cryptographic secure aggregation~\\cite{bonawitz2017} protects update confidentiality but not integrity and is orthogonal to our design. '
    + R('Table~\\ref{tab:novelty} summarizes the differences: to our knowledge no prior scheme makes Byzantine-robust aggregation aware of the sparsity pattern of compressed updates, and none reports node deaths together with robustness to poisoning.') + NOVELTY
    + '\n\n' + R('\\textit{Trust models, state estimation and intrusion detection in WSNs.} Reputation and trust frameworks let nodes rate their neighbours from observed behaviour, for example with a Bayesian beta reputation system~\\cite{ganeriwal2008}; best practices for such systems are reviewed in~\\cite{lopez2010}. Probabilistic state estimation detects faulty sensors from the residuals of a model of the monitored process, for instance trust-enhanced distributed Kalman filtering~\\cite{shaheen2025}, and intrusion detection systems flag anomalous node or traffic behaviour, often with context such as location or protocol state~\\cite{butun2014}. These methods judge \\emph{nodes} or \\emph{readings} and are complementary to EASFL. EASFL protects the \\emph{training process}: its reputation scores are computed from model updates, its direction test runs on sparsified updates without discarding honest ones that carry different coordinates, and its output is the anomaly detector that such systems could also use. A trust score from an existing IDS could be fed into EASFL\'s CH score or quarantine in place of, or in addition to, the update-based suspicion score.'))

SWITCH = R(r"""\textit{Choosing the CH policy as the network ages.} Score-based CH selection keeps the CH role on high-energy nodes, which equalizes residual energy: nodes die late but close together. LEACH rotation spreads the role evenly from the first round, so weak nodes die early but the last ones last longer. We tested rules that start with score-based selection and switch to LEACH rotation once a fraction of nodes is dead or the mean residual energy falls below a fraction of its initial value (energy-only simulation, 20 seeds, paired Wilcoxon tests; Table~\ref{tab:switch}). Switching after nodes start dying changes little, because energies are already equalized. Switching when the mean residual energy falls below 25\% gives the most node-rounds of service (+1.0\%, $p<0.001$) and a later HND (+18 rounds) at the cost of 19 rounds of FND. Pure LEACH rotation gives the longest LND (+61 rounds) but loses 397 rounds of FND, and a CH cooldown loses both FND and node-rounds. Our guidance is therefore: keep score-based selection when full coverage matters (FND, monitoring without gaps); switch at about 25\% residual energy when total service matters; use rotation from the start only when the last surviving nodes matter more than early coverage. The CH fraction $p$ should be chosen with the sweep in Supplementary Table~S1, where $p=0.05$--$0.2$ left EASFL's FND ahead of every baseline.""") + r"""

\begin{table}[t]
\caption{""" + R('Effect of the CH Policy on Lifetime (Synthetic Network, 20 Seeds; Differences to Score-Based Selection, Paired)') + r"""}\label{tab:switch}
\centering\scriptsize\setlength{\tabcolsep}{3pt}\revtab
\begin{tabular}{@{}lcccc@{}}
\toprule
CH policy & FND & HND & LND & Node-rounds\\
\midrule
Score-based (EASFL) & 558 & 660 & 742 & 66{,}813\\
LEACH rotation & $-397$ & $+43$ & $+61$ & $-551$\\
Score + cooldown & $-227$ & $-34$ & $+33$ & $-4{,}094$\\
Switch at 10\% dead & 0 & $+5$ & $-6$ & $+98$\\
Switch at 50\% dead & 0 & 0 & $-4^{\ast}$ & $-36$\\
Switch at $\bar E<50\%$ & $-138$ & $+28$ & $-12^{\ast}$ & $+353$\\
Switch at $\bar E<25\%$ & $-19$ & $+18$ & $-13$ & $+674$\\
\bottomrule
\end{tabular}

\vspace{1mm}\parbox{\columnwidth}{\scriptsize All differences are significant at $p<0.05$ (Wilcoxon) except those marked $^\ast$. Node-rounds: sum over rounds of the number of alive nodes.}
\end{table}
"""
rep('\\textit{Threat model.} ', SWITCH + '\n\n\\textit{Threat model.} ')


# ======================= stage 6: physically recorded, labelled anomalies (E13: AE-5, R1-8, R2-4) =======================
REAL = r"""
\subsection{""" + R('Detection of Physically Recorded Anomalies') + r"""}\label{sec:real}
""" + R(r"""The Intel Lab validation uses injected faults. To test the detector on anomalies that were physically produced and labelled by others, we used the Labelled Wireless Sensor Network Data Repository~\cite{suthaharan2010}: two deployments of four TelosB motes each (single-hop and multi-hop) that recorded temperature and humidity every 5\,s for 6\,h, during which the dataset's authors released steam from hot water near some motes and labelled every affected reading. The data contain four such events on four motes. Each mote is one FL client and readings are grouped into windows of 10 samples (50\,s); light and voltage are not recorded, so those channels are held constant and their features carry no information. Every detector is trained only on the first 60\% of each mote's windows, with all windows near a real event removed and synthetic faults injected exactly as in Section~\ref{sec:intel} (restricted to the two measured channels). The real events are never seen in training. The test set holds all 33 real-event windows and the 1{,}510 normal windows of the last 40\% of each trace. All FL schemes run 200 rounds with EASFL's settings (10 seeds; per-node thresholds).

Table~\ref{tab:real} shows the result. EASFL-M detected all four events in every seed, at 1.7 false alarms per hour of normal operation, at least as well as matched FedAvg (3.7 of 4 events on average) and better than full-precision FedAvg-LEACH and centralized training. At the level of single windows, however, every supervised detector flags only part of each event (window recall 0.24--0.48, F1 0.21--0.37): the steam events raise temperature and humidity together over several minutes, whereas the injected training faults mostly affect one channel. An unsupervised isolation forest fitted on each mote's own normal windows flags every event window (recall 1.00) at a higher false-alarm rate (3.0 per hour). Two conclusions follow. First, training on injected faults transfers to real events at the event level but only partly at the window level, consistent with the unseen-fault test of Section~\ref{sec:extra}. Second, an on-node unsupervised score is a useful complement to the federated detector for large, multi-channel events. With four events on eight motes this is a check on real anomalies, not a benchmark; a larger labelled deployment remains desirable.""") + r"""

\begin{table}[t]
\caption{""" + R('Detection of Real Steam Events (LWSNDR, 8 TelosB Motes, 4 Events; Trained on Injected Faults Only; 10 Seeds)') + r"""}\label{tab:real}
\centering\scriptsize\setlength{\tabcolsep}{2.4pt}\revtab
\begin{tabular}{@{}lcccccc@{}}
\toprule
Method & F1 & Prec. & Rec. & AUC & Events & FA/h\\
\midrule
""" + 'EASFL-M & \\pms{0.31}{0.06} & 0.29 & 0.37 & 0.68 & 4.0/4 & 1.7\\\\\nEASFL (dense filter) & \\pms{0.33}{0.05} & 0.28 & 0.42 & 0.70 & 3.9/4 & 1.8\\\\\nFedAvg, matched & \\pms{0.29}{0.08} & 0.28 & 0.34 & 0.64 & 3.7/4 & 1.7\\\\\nFedAvg-LEACH & \\pms{0.24}{0.07} & 0.23 & 0.32 & 0.62 & 3.6/4 & 2.1\\\\\nCentralized & \\pms{0.21}{0.08} & 0.19 & 0.24 & 0.60 & 3.5/4 & 1.8\\\\\nLocal only & \\pms{0.37}{0.05} & 0.30 & 0.48 & 0.74 & 3.8/4 & 1.9\\\\\nIsolation forest (unsup.) & \\pms{0.51}{0.01} & 0.35 & 1.00 & 0.98 & 4.0/4 & 3.0\\\\' + r"""
\bottomrule
\end{tabular}

\vspace{1mm}\parbox{\columnwidth}{\scriptsize Window-level metrics on 33 event and 1{,}510 normal windows; Events: real events with at least one flagged window (mean over seeds); FA/h: false alarms per hour of normal operation.}
\end{table}
"""
rep('\n\\subsection{' + R('Measurement on IEEE 802.15.4 Hardware') + '}\\label{sec:testbed}',
    REAL + '\n\\subsection{' + R('Measurement on IEEE 802.15.4 Hardware') + '}\\label{sec:testbed}')
rep('the Intel Lab experiment is therefore a semi-synthetic validation, and validation on traces with expert-labelled faults remains future work.',
    'the Intel Lab experiment is therefore a semi-synthetic validation. ' + R('On physically recorded, labelled steam events (Section~\\ref{sec:real}) EASFL-M detected every event but flagged only part of each event window by window, and an unsupervised on-node score complemented it.'))


# ======================= stage 7: page budget - move secondary material to the supplementary file =======================
SUPP = []
def move(start, end, pointer, title):
    """Cut src[start:end) (end marker kept) into the supplement and leave `pointer` in its place."""
    global src, n_ok
    i = src.index(start); j = src.index(end, i)
    SUPP.append((title, src[i:j])); src = src[:i] + pointer + src[j:]; n_ok += 1

move('\\subsection{Packet Loss}', '\\subsection{Validation on the Intel Lab Deployment}',
     '\\subsection{' + R('Packet Loss, Workload Sensitivity and Sampling Rate') + '}\n'
     + R('With packet-loss probabilities up to 0.3, EASFL\'s global F1 stays between 0.69 and 0.72, whereas FedAvg-LEACH falls from 0.69 to 0.65. Across non-IID degree ($\\alpha=0.1$--10), fault magnitude ($0.5$--$1.5\\times$) and anomaly rate ($0.5$--$2\\times$), personalized EASFL matches or exceeds the centralized model except for the smallest faults, where all methods reach F1 $\\approx0.54$. Centralized collection outlives EASFL in HND only at very low reporting rates ($\\le$10 samples per round), and its FND remains far shorter. Details are in Supplementary Section~S-B.') + '\n\n',
     'Packet Loss, Workload Sensitivity and Sampling Rate')
i = src.index('\\caption{\\rev{Energy by Protocol Component'); i = src.rindex('\\begin{table}', 0, i)
j = src.index('\\end{table}', i) + len('\\end{table}')
SUPP.append(('Energy by Protocol Component', src[i:j])); src = src[:i] + src[j:]; n_ok += 1
src = src.replace('Table~\\ref{tab:breakdown}', 'Supplementary Table~S5')
i = src.index('\\caption{\\rev{Effect of the CH Policy on Lifetime'); i = src.rindex('\\begin{table}', 0, i)
j = src.index('\\end{table}', i) + len('\\end{table}')
SUPP.append(('Effect of the CH Policy on Lifetime', src[i:j])); src = src[:i] + src[j:]; n_ok += 1
src = src.replace('Table~\\ref{tab:switch}', 'Supplementary Table~S6')

i = src.index('\\caption{\\rev{Network-Level F1, Coverage and Cumulative Detection'); i = src.rindex('\\begin{table}', 0, i)
j = src.index('\\end{table}', i) + len('\\end{table}')
SUPP.append(('Lifetime F1', src[i:j])); src = src[:i] + src[j:]; n_ok += 1
src = src.replace('Table~\\ref{tab:lifef1}', 'Supplementary Table~S7')
i = src.index('\\textit{Detector choice and model size.}'); j = src.index('\\textit{Sensitivity.} Supplementary', i)
SUPP.append(('Detector choice', src[i:j]))
src = src[:i] + ('\\textit{Detector choice, model size and thresholds.} The 417-parameter MLP on summary features outperformed an isolation forest, a GRU, a 1D-CNN classifier and a 1D-CNN autoencoder on the same data (Supplementary Table~S3); larger federated MLPs keep EASFL-M\'s robustness but multiply radio energy, so model size must be chosen against the energy budget. Thresholds chosen on a held-out validation split lower per-node F1 from 0.830 to 0.769, and personalization costs about 1.2\\,mJ per node once (Supplementary Section~S-D).\n\n') + src[j:]; n_ok += 1
# supplement body: strip revision colour and float placement so it compiles in the supplementary file
sup = []
for title, body in SUPP:
    b = body.replace('Section~\\ref{sec:model}', 'Section~III of the main paper').replace('Section~\\ref{sec:matched}', 'the matched-baseline section of the main paper').replace('\\revtab', '').replace('[t]', '[h]').replace('figure*', 'figure').replace('0.95\\textwidth', '\\columnwidth')
    b = b.replace('\\label{tab:breakdown}', '\\label{tab:s5}').replace('\\label{tab:switch}', '\\label{tab:s6}')
    sup.append(b)
open('supp_moved.tex', 'w').write('% generated by apply_revision.py (stage 7): material moved from the main text\n'
    '\\section*{S-B. Packet Loss, Workload Sensitivity and Sampling Rate}\n' + sup[0].replace('\\subsection{', '\\subsubsection*{')
    + '\n\\section*{S-C. Energy Breakdown, CH-Policy Switching and Lifetime Detection}\n\\setcounter{table}{4}\n' + sup[1] + '\n' + sup[2] + '\n' + sup[3].replace('\\label{tab:lifef1}', '\\label{tab:s7}') + '\n'
    + '\\section*{S-D. Detector Choice, Model Size and Thresholds}\n' + sup[4] + '\n')

# ======================= stage 8: figures regenerated with 20 seeds (10 for poisoning, EASFL-M added) (R1-6, R1-9) =======================
for old, new in [
    ('\\caption{F1 of the global model during training (mean $\\pm$ std, 5 seeds). FedAvg-flat degrades as its nodes die.}',
     '\\caption{F1 of the global model during training (mean $\\pm$ std, \\rev{20 seeds; EASFL-M added}). FedAvg-flat degrades as its nodes die.}'),
    ('\\caption{Alive nodes per round (mean of 5 seeds, 60 raw samples per node-round for centralized collection).}',
     '\\caption{Alive nodes per round (mean of \\rev{20} seeds, 60 raw samples per node-round for centralized collection).}'),
    ('\\caption{Ablation: contribution of each EASFL component to lifetime (mean $\\pm$ std).}',
     '\\caption{Ablation: contribution of each EASFL component to lifetime (mean $\\pm$ std\\rev{, 20 seeds}).}'),
    ('\\caption{Global-model F1 versus fraction of malicious nodes for four poisoning attacks (mean of 5 seeds).}',
     '\\caption{Global-model F1 versus fraction of malicious nodes for four poisoning attacks (mean of \\rev{10} seeds\\rev{; EASFL-M added}).}'),
]:
    assert src.count(old) == 1, old[:50]
    src = src.replace(old, new); n_ok += 1
old = 'because selecting a single update discards the information carried by heterogeneous nodes.'
new = old + ' \\rev{Fig.~\\ref{fig:pois} also shows the limit of the defence: at 30\\% malicious nodes, beyond the 20\\% design point, EASFL and EASFL-M fall to 0.44--0.50 under the scaled sign flip and to 0.36--0.45 under the label flip, with large spread across seeds; neither then beats the coordinate median under the sign flip (0.53), and EASFL-M also falls below median and trimmed mean under the label flip (0.41).}'
assert src.count(old) == 1
src = src.replace(old, new); n_ok += 1

open('main_revised.tex', 'w').write(src)
open('main_clean.tex', 'w').write('\\def\\cleanversion{1}\n\\input{main_revised.tex}\n')
print('edits applied:', n_ok + 1)
