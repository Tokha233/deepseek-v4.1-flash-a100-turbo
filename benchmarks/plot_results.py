"""Plot retained windows and both community-protocol repeats (requires matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parent


def main() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.edgecolor': '#cbd5e1', 'text.color': '#172554', 'axes.labelcolor': '#334155',
        'xtick.color': '#475569', 'ytick.color': '#475569', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    runs = [('C32 / output 1,024', 'short-c32-o1024', '#64748b'),
        ('C128 / output 1,024', 'short-c128-o1024', '#2563eb'),
        ('C256 / output 1,024', 'short-c256-o1024', '#0891b2'),
        ('C128 / output 128', 'short-c128-o128', '#d97706')]
    for label, name, color in runs:
        data = json.loads((ROOT / 'results/round14' / name / 'measurement-summary.json').read_text())
        axes[0].plot(range(30, 181, 30), data['all_30s_output_tps'], marker='o', lw=2,
            ms=4, color=color, label=f"{label}: {data['output_tokens_per_second']:,.0f}")
    axes[0].set(title='Short coding · sustained serving', xlabel='Seconds into fixed measurement window',
        ylabel='Output tokens / second', ylim=(0, 3500), xticks=list(range(30, 181, 30)))
    axes[0].legend(loc='lower right', frameon=False, fontsize=9)
    axes[0].grid(axis='y', alpha=.18)
    axes[0].text(0, -.24, '8×A800 · effort100 · T1 · unique cache salts\n60 s warmup + 180 s window · all 30 s bins shown',
        transform=axes[0].transAxes, fontsize=9, color='#64748b')
    for i, mode in enumerate(['off', 'high']):
        values = [json.loads((ROOT / f'results/round14/community-{mode}-repeat{r}/report.json').read_text())[
            'load_output_tokens_per_second'] for r in [1, 2]]
        ours = sum(values) / len(values)
        reference = {'off': 1379.01, 'high': 1039.91}[mode]
        axes[1].barh(i + .17, ours, height=.30, color='#2563eb',
            label='This deployment · 8×A800' if i == 0 else None)
        axes[1].scatter(values, [i + .17] * 2, color='white', s=22, edgecolors='#172554', zorder=3)
        axes[1].barh(i - .17, reference, height=.30, color='#94a3b8',
            label='keplerzip report · 8×A100' if i == 0 else None)
        axes[1].text(ours + 35, i + .17, f'{ours:,.0f}  (+{(ours/reference-1)*100:.1f}%)', va='center', fontsize=9)
        axes[1].text(reference + 35, i - .17, f'{reference:,.0f}', va='center', fontsize=9, color='#64748b')
    axes[1].set(title='Community protocol · C32', xlabel='Output tokens / second',
        yticks=[0, 1], yticklabels=['Thinking off', 'High / effort75'], ylim=(-.6, 1.65))
    axes[1].set_xlim(0, axes[1].get_xlim()[1] * 1.26)
    axes[1].legend(loc='upper left', frameon=False, fontsize=9)
    axes[1].grid(axis='x', alpha=.15)
    axes[1].set_axisbelow(True)
    axes[1].text(0, -.24, 'T0 · fixed 1,024 output · 200 requests, full batch + tail\nBlue: mean of 2 runs; dots: both runs. Different hardware/config.',
        transform=axes[1].transAxes, fontsize=9, color='#64748b')
    for ax in axes:
        ax.tick_params(length=0, pad=8)
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    fig.suptitle('DeepSeek-V4.1-Flash on Ampere', fontsize=16, fontweight='bold', x=.02, ha='left')
    (ROOT / 'figures').mkdir(exist_ok=True)
    for suffix in ['svg', 'png']:
        fig.savefig(ROOT / f'figures/short-and-community.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
