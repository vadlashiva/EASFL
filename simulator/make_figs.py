"""Publication figures from results_v2.json -> figs/*.pdf and figs/*.png"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('results_v2.json'))
os.makedirs('figs', exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # fixed categorical order
MK = ['o', 's', '^', 'D', 'v', 'P']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6,
                     'lines.linewidth': 2, 'legend.frameon': False, 'savefig.dpi': 300})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'figs/{name}.pdf'); fig.savefig(f'figs/{name}.png'); plt.close(fig)


# Fig 1: alive nodes vs rounds
if 'life' in R:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for j, (sc, lab) in enumerate([('CL', 'Centralized (LEACH)'), ('FL-flat', 'FedAvg-flat'),
                                   ('FL-LEACH', 'FedAvg-LEACH'), ('EASFL', 'EASFL (proposed)')]):
        rows = [r for r in R['life'] if r['scheme'] == sc and r['kw'] == {}]
        T = max(len(r['alive']) for r in rows)
        A = np.array([np.pad(r['alive'], (0, T - len(r['alive']))) for r in rows]).mean(0)
        ax.plot(np.arange(1, T + 1), A, color=C[j], label=lab)
    ax.set_xlabel('Round'); ax.set_ylabel('Alive nodes'); ax.set_ylim(0, 102); ax.legend(fontsize=7)
    save(fig, 'fig_alive')

    # Fig 5: ablation of lifetime
    var = [('EASFL', {}, 'EASFL (full)'), ('EASFL', {'ea_ch': False}, 'w/o energy-aware CH'),
           ('EASFL', {'psel': False}, 'w/o participant sel.'), ('EASFL', {'compress': False}, 'w/o compression'),
           ('FL-LEACH', {}, 'FedAvg-LEACH')]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y = np.arange(len(var)); h = 0.38
    for k, (m, lab, col) in enumerate([('fnd', 'First node dead', C[0]), ('hnd', 'Half nodes dead', C[1])]):
        v = [np.mean([r[m] for r in R['life'] if r['scheme'] == sc and r['kw'] == kw]) for sc, kw, _ in var]
        e = [np.std([r[m] for r in R['life'] if r['scheme'] == sc and r['kw'] == kw]) for sc, kw, _ in var]
        ax.barh(y + (k - 0.5) * h, v, h * 0.92, xerr=e, color=col, label=lab, error_kw=dict(lw=0.8))
    ax.set_yticks(y); ax.set_yticklabels([v[2] for v in var], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('Rounds'); ax.legend(fontsize=7, loc='lower right'); ax.grid(axis='y', visible=False)
    save(fig, 'fig_ablation')

# Fig 2: convergence
if 'learn' in R:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for j, (sc, lab) in enumerate([('FL-flat', 'FedAvg-flat'), ('FL-LEACH', 'FedAvg-LEACH'), ('EASFL', 'EASFL')]):
        rows = [r for r in R['learn'] if r['scheme'] == sc and r['kw'] == {}]
        x = [c[0] for c in rows[0]['curve']]
        f = np.array([[c[1] for c in r['curve']] for r in rows])
        ax.plot(x, f.mean(0), color=C[j + 1], label=lab)
        ax.fill_between(x, f.mean(0) - f.std(0), f.mean(0) + f.std(0), color=C[j + 1], alpha=0.12, lw=0)
    cl = np.mean([r['final']['global']['f1'] for r in R['central']])
    ax.axhline(cl, color='#555', ls='--', lw=1); ax.text(5, cl + 0.01, 'Centralized', fontsize=7, color='#333')
    ax.set_xlabel('Round'); ax.set_ylabel('F1 (global model)'); ax.legend(fontsize=7, loc='lower right')
    save(fig, 'fig_convergence')

# Fig 3: poisoning
if 'attack' in R:
    D = [('FL-LEACH', {}, 'FedAvg (mean)'), ('FL-LEACH', {'agg': 'median'}, 'Median'),
         ('FL-LEACH', {'agg': 'trimmed'}, 'Trimmed mean'), ('FL-LEACH', {'agg': 'krum'}, 'Krum'),
         ('EASFL', {}, 'EASFL')]
    atts = [('signflip', 'Scaled sign flip'), ('stealth', 'Stealth sign flip'),
            ('labelflip', 'Label flip'), ('gauss', 'Gaussian noise')]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    for ax, (a, title) in zip(axs, atts):
        for j, (sc, kw, lab) in enumerate(D):
            xs, ys = [], []
            for m in [0.0, 0.1, 0.2, 0.3]:
                rows = [r for r in R['attack'] if r['scheme'] == sc and r['kw']['mal_frac'] == m
                        and (m == 0 or r['kw'].get('attack') == a)
                        and {k: v for k, v in r['kw'].items() if k not in ('mal_frac', 'attack')} == kw]
                xs.append(m * 100); ys.append(np.mean([r['final']['global']['f1'] for r in rows]))
            ax.plot(xs, ys, color=C[j], marker=MK[j], ms=4, lw=1.5, label=lab)
        ax.set_title(title, fontsize=8); ax.set_xlabel('Malicious nodes (%)'); ax.set_xticks([0, 10, 20, 30])
    axs[0].set_ylabel('F1 (global model)'); axs[0].set_ylim(0, 0.8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('figs/fig_poisoning.pdf'); fig.savefig('figs/fig_poisoning.png'); plt.close(fig)

# Fig 4: generator sensitivity
if 'gen' in R:
    def series(param, vals):
        out = {}
        for v in vals:
            g = {param: v}
            if (param, v) in [('alpha', 0.5), ('amp_scale', 1.0), ('rate_scale', 1.0)]:
                cen, fl = R['central'], [r for r in R['learn'] if r['kw'] == {}]
            else:
                cen = [r for r in R['gen_central'] if r['gen'] == g]; fl = [r for r in R['gen'] if r['gen'] == g]
            out.setdefault('Centralized', []).append(np.mean([r['final']['global']['f1'] for r in cen]))
            for sc, lab in [('FL-LEACH', 'FedAvg-LEACH'), ('EASFL', 'EASFL')]:
                rr = [r for r in fl if r['scheme'] == sc]
                out.setdefault(lab, []).append(np.mean([r['final']['global']['f1'] for r in rr]))
                if sc == 'EASFL':
                    out.setdefault('EASFL + personalisation', []).append(np.mean([r['final']['ft+thr']['f1'] for r in rr]))
        return out
    panels = [('alpha', [0.1, 0.3, 0.5, 1.0, 10.0], r'Non-IID: Dirichlet $\alpha$', True),
              ('amp_scale', [0.5, 0.75, 1.0, 1.5], 'Fault magnitude (×)', False),
              ('rate_scale', [0.5, 1.0, 2.0], 'Anomaly rate (×)', False)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (p, vals, xl, logx) in zip(axs, panels):
        s = series(p, vals)
        for j, (lab, ys) in enumerate(s.items()):
            ax.plot(vals, ys, color=['#555555', C[1], C[2], C[0]][j], ls='--' if j == 0 else '-',
                    marker=MK[j], ms=4, lw=1.5, label=lab)
        if logx: ax.set_xscale('log')
        ax.set_xlabel(xl)
    axs[0].set_ylabel('F1')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig('figs/fig_sensitivity.pdf'); fig.savefig('figs/fig_sensitivity.png'); plt.close(fig)
print('figures written to figs/')
