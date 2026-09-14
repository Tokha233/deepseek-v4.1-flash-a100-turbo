"""Render the single-node concurrency curve and measured incremental ablation."""

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parent


def main() -> None:
    data = ROOT / 'results/round15'
    concurrency = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    throughput = [json.loads((data / f'full-c{c}/measurement-summary.json').read_text())[
        'output_tokens_per_second'] for c in concurrency]
    stages = ['base', 'dense', 'full']
    values = [[json.loads((data / f'ablation-{s}-repeat{r}/measurement-summary.json').read_text())[
        'output_tokens_per_second'] for r in [1, 2]] for s in stages]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.edgecolor': '#cbd5e1', 'text.color': '#172554', 'axes.labelcolor': '#334155',
        'xtick.color': '#475569', 'ytick.color': '#475569'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    axes[0].plot(concurrency, throughput, color='#2563eb', marker='o', lw=2.5, ms=5)
    axes[0].fill_between(concurrency, throughput, color='#2563eb', alpha=.07)
    axes[0].set(xscale='log', title='Sustained throughput', xlabel='Client concurrency',
        ylabel='Output tokens / second', ylim=(0, max(throughput) * 1.20))
    axes[0].set_xticks(concurrency, labels=[str(c) for c in concurrency])
    for c, t in zip(concurrency, throughput):
        axes[0].annotate(f'{t:,.0f}', (c, t), xytext=(0, 9), textcoords='offset points',
            ha='center', fontsize=9, color='#1d4ed8')
    labels = ['Marlin + NCCL', '+ Dense BF16', '+ Custom EP8 AG/RS']
    for i, runs in enumerate(values):
        avg = mean(runs)
        axes[1].bar(i, avg, width=.58, color=['#94a3b8', '#0891b2', '#2563eb'][i])
        axes[1].scatter([i - .05, i + .05], runs, s=27, color='white', edgecolors='#172554', zorder=3)
        change = '' if not i else f'\n{(avg / mean(values[i - 1]) - 1) * 100:+.1f}%'
        axes[1].text(i, max(runs) + max(map(mean, values)) * .035, f'{avg:,.0f}' + change,
            ha='center', fontsize=10)
    axes[1].set(title='Incremental ablation · C128', xticks=range(3), xticklabels=labels,
        ylim=(0, max(map(max, values)) * 1.27))
    for ax in axes:
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    fig.suptitle('DeepSeek-V4.1-Flash  |  8 x A800  |  1M context',
        x=.02, ha='left', fontsize=16, fontweight='bold')
    fig.supxlabel('Short coding · effort100 · T1 · fixed 1,024 output · 60 s warmup + 180 s measurement\n'
        'Ablation: mean of two runs, both shown as dots. DSpark5 / CUDA Graph / EPLB retained in every stage.',
        fontsize=9, color='#64748b')
    for extension in ['svg', 'png']:
        path = ROOT / f'figures/single-node.{extension}'
        fig.savefig(path, dpi=180, facecolor='white')
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
