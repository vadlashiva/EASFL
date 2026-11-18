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

open('main_revised.tex', 'w').write(src)
open('main_clean.tex', 'w').write('\\def\\cleanversion{1}\n\\input{main_revised.tex}\n')
print('edits applied:', n_ok + 1)
