"""Rebuild the article overview and the source-backed context-length figure.

Requires Pillow, NumPy and Matplotlib. Table 9 data are transcribed from
https://arxiv.org/html/2603.08274v1 (not Table 7/8 accuracy-optimized rates).
"""
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONT_DIR = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts'

def overview():
    regular, bold = FONT_DIR / 'msyh.ttc', FONT_DIR / 'msyhbd.ttc'
    def font(size, weight=False):
        return ImageFont.truetype(str(bold if weight else regular), size)
    im = Image.new('RGB', (1600, 1000), '#F5F3EC')
    draw = ImageDraw.Draw(im)
    ink, muted, teal = '#20343D', '#5E7077', '#087F78'
    def text(x, y, value, size=28, weight=False, color=ink):
        draw.text((x, y), value, font=font(size, weight), fill=color)
    draw.rounded_rectangle((72, 68, 132, 80), radius=6, fill=teal)
    text(72, 110, 'AI Coding 效率提升的两个层次', 50, True)
    text(75, 192, '先优化一次任务的效率，再在多个任务组的实践中提升人的能力。', 25, color=muted)
    draw.rounded_rectangle((72, 275, 1528, 640), radius=24, fill='#D0E5E1')
    text(108, 310, '01  任务内的效率提升', 38, True)
    text(108, 374, '把一次任务做得更快、更稳', 30)
    columns = [(108, '明确目标与计划', '范围 · 约束 · 验收标准'),
               (595, '组织执行与上下文', '有效信息 · 小步执行'),
               (1082, '验证结果并复用', '运行证据 · 提交 · skill')]
    for x, title, detail in columns:
        text(x, 464, title, 29, True)
        text(x, 518, detail, 23, color=muted)
    text(108, 584, '把计划、执行、验证与复用组织成可重复的过程。', 25, color=muted)
    draw.line((790, 651, 790, 677), fill=teal, width=3)
    draw.polygon([(790, 685), (782, 673), (798, 673)], fill=teal)
    draw.rounded_rectangle((72, 700, 1528, 893), radius=24, fill='#E6E2D4')
    text(108, 730, '02  人的提升', 36, True)
    text(600, 736, '扩大能理解、能验证、能负责的范围', 29, True)
    text(108, 812, '跨任务积累经验和判断', 26)
    text(600, 812, '保留练习 · 偿还认知债务 · 记录意图', 26, color=muted)
    text(75, 935, '交付循环提高当下效率，学习循环支撑长期能力。', 25, color=muted)
    im.save(OUT / 'efficiency-layers.png')

# Percentages; all 11 models with 32K, 128K and 200K entries in Table 9.
FABRICATION = [
    ('GLM 4.6', [7.04, 13.75, 69.53]),
    ('Qwen3 Next 80B-A3B', [7.04, 7.99, 10.25]),
    ('Qwen3 Coder 480B-A35B', [12.20, 16.88, 27.52]),
    ('GLM 4.7', [16.27, 10.42, 46.73]),
    ('Qwen3 235B-A22B', [19.84, 22.08, 33.15]),
    ('Qwen3 235B-A22B FP8', [23.31, 21.81, 32.94]),
    ('Qwen3 4B 2507', [25.30, 29.79, 26.02]),
    ('Llama 4 Maverick', [28.08, 38.82, 43.29]),
    ('Qwen3 30B-A3B 2507', [36.11, 35.07, 41.04]),
    ('Qwen3 Coder 30B-A3B', [40.18, 43.82, 47.21]),
    ('Llama 4 Scout', [44.44, 52.22, 52.09]),
]

def context_chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    for name in ('msyh.ttc', 'msyhbd.ttc'):
        font_manager.fontManager.addfont(str(FONT_DIR / name))
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'svg.fonttype': 'path',
                         'axes.unicode_minus': False, 'font.size': 16})
    values = np.array([row[1] for row in FABRICATION])
    fig = plt.figure(figsize=(16, 11), dpi=120, facecolor='#F5F3EC')
    fig.text(.055, .925, '长上下文的代价：最佳编造率', fontsize=29, weight='bold', color='#20343D')
    fig.text(.055, .875, '三档均完成测试的 11 个模型｜数值越低越好', fontsize=17, color='#5E7077')
    ax = fig.add_axes([.345, .19, .53, .62])
    cmap = LinearSegmentedColormap.from_list('risk', ['#F7EED8', '#E7B47B', '#B65F3D', '#763322'])
    mesh = ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    ax.set_xticks([0, 1, 2], ['32K tokens', '128K tokens', '200K tokens'])
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=15, labelsize=18)
    ax.set_yticks(range(len(FABRICATION)), [row[0] for row in FABRICATION])
    ax.tick_params(axis='y', length=0, pad=15, labelsize=15)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which='minor', color='#F5F3EC', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(len(FABRICATION)):
        for col in range(3):
            value = values[row, col]
            ax.text(col, row, f'{value:.2f}%', ha='center', va='center', fontsize=16,
                    color='white' if value >= 60 else '#20343D', weight='bold')
    cax = fig.add_axes([.905, .19, .019, .62])
    bar = fig.colorbar(mesh, cax=cax, ticks=[0, 25, 50, 75, 100])
    bar.set_label('编造率（%）', fontsize=14, labelpad=12)
    bar.outline.set_visible(False)
    fig.text(.055, .115, '口径：缺失信息探测题的编造率；每档分别取所测温度与硬件组合中的最低值。', fontsize=13, color='#5E7077')
    fig.text(.055, .078, '范围：合成英文文档问答、开放权重模型；不代表日常回答或代码生成的总体错误率。', fontsize=13, color='#5E7077')
    fig.text(.055, .039, '来源：arXiv:2603.08274v1，Table 9  ·  https://arxiv.org/html/2603.08274v1', fontsize=12, color='#5E7077')
    fig.savefig(OUT / 'context-fabrication-rates.png', dpi=120, facecolor=fig.get_facecolor())
    svg = OUT / 'context-fabrication-rates.svg'
    fig.savefig(svg, facecolor=fig.get_facecolor())
    # Matplotlib emits path lines with trailing spaces; normalize for git diff --check.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)

if __name__ == '__main__':
    overview()
    context_chart()
