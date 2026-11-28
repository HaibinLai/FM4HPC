# Builds overview.html from FINDINGS.csv and lean_files.csv (run from the repo root:
#   python3 operators/magma/viz/make_overview.py)
import csv, collections, html, json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
F = [r for r in csv.DictReader(open(os.path.join(ROOT, '..', 'FINDINGS.csv'), newline='')) if r['id'] != 'M244']
L = [l.strip().split(',') for l in open(os.path.join(ROOT, 'lean_files.csv')) if l.strip()]
L = [(p.replace('./OptimiKahan/', '').replace('./', ''), int(a), int(b), int(c)) for p, a, b, c, _ in L if not p.endswith('lakefile.lean')]
ZH = {  # Chinese labels for the categories
 'mixed-precision-IR': '混合精度迭代细化', 'sparse-control/IO/format': '稀疏：格式/IO/工具', 'batched/vbatched': '批量 / 可变尺寸批量',
 'dense-eigen': '稠密特征值', 'sparse-solvers': '稀疏迭代求解器', 'sparse-blas': '稀疏 BLAS', 'magmablas-kernels': 'magmablas 内核',
 'dense-QR/least-squares': '稠密 QR / 最小二乘', 'sparse-preconditioners': '稀疏预条件', 'multi-GPU/out-of-core': '多卡 / 核外',
 'dense-LU': '稠密 LU', 'dense-LDL/Hermitian-indefinite': '稠密 LDL^H / 不定', 'dense-reduction': '稠密约化', 'dense-Cholesky': '稠密 Cholesky',
 'infrastructure': '基础设施', 'dense-inverse/triangular': '稠密求逆 / 三角', 'dense-SVD': '稠密 SVD',
 'silent-wrong-result': '静默错误结果', 'missing-check': '缺少检查', 'memory-safety': '内存安全', 'status-lost-or-overwritten': '状态丢失或被覆盖',
 'numerical-method': '数值方法', 'wrong-status-value': '状态值错误', 'workspace-sizing': '工作区大小', 'build-or-API': '构建 / API / 文档',
 'wrong-argument-number': '参数编号错误', 'concurrency-race': '并发竞争', 'integer-overflow': '整数溢出', 'launch-config': '启动配置', 'performance-only': '仅性能',
 'crash-or-memory-corruption': '崩溃或内存损坏', 'misleading-status': '误导性的状态', 'rejected-valid-call': '拒绝合法调用', 'cosmetic-message': '仅提示信息', 'performance': '性能',
 'normal-input': '普通输入', 'singular-or-indefinite': '奇异 / 不定', 'caller-misuse-or-stale-state': '调用方误用 / 残留状态', 'edge-size': '边界尺寸', 'non-finite-input': '非有限值输入',
 'extreme-scaling': '极端缩放', 'large-size': '大尺寸', 'allocation-failure': '分配失败', 'build-option-or-backend': '构建选项 / 后端', 'multi-gpu': '多卡'}
def count(k): return collections.Counter(r[k] for r in F).most_common()
n = len(F); gpu = sum(1 for r in F if r['evidence'] == 'gpu-confirmed'); silent = sum(1 for r in F if r['impact'] == 'silent-wrong-result')
lean_link = sum(1 for r in F if r['lean_theorems'].strip() not in ('', '—', '-'))
lines = sum(x[1] for x in L); thms = sum(x[2] for x in L); defs = sum(x[3] for x in L)
def group(p):
    if p.startswith('Magma/'): return 'MAGMA 2.10.0'
    if p in ('IrsContract.lean', 'IrDriver.lean', 'MaxFold.lean'): return '跨库 IR 接受逻辑'
    return '低精度模拟高精度'
G = collections.OrderedDict()
for p, a, b, c in L: g = G.setdefault(group(p), [0, 0, 0]); g[0] += 1; g[1] += a; g[2] += b
charts = [
 ('c1', '按子系统：缺陷集中在哪里', [(ZH.get(k, k), v) for k, v in count('subsystem')], '条'),
 ('c2', '按缺陷类型', [(ZH.get(k, k), v) for k, v in count('defect_class')], '条'),
 ('c3', '按后果', [(ZH.get(k, k), v) for k, v in count('impact')], '条'),
 ('c4', '按触发条件', [(ZH.get(k, k), v) for k, v in count('trigger')], '条'),
 ('c5', 'Lean 代码：每个文件的行数', sorted([(p.replace('.lean', ''), a) for p, a, b, c in L], key=lambda x: -x[1]), '行'),
]
thm_of = {p.replace('.lean', ''): b for p, a, b, c in L}
data = json.dumps({'charts': [{'id': i, 'rows': r, 'unit': u} for i, t, r, u in charts], 'thm': thm_of}, ensure_ascii=False)
def table(rows, unit, extra=None):
    h = '<table><thead><tr><th>类别</th><th>数量（' + unit + '）</th>' + ('<th>定理</th>' if extra else '') + '</tr></thead><tbody>'
    for k, v in rows: h += '<tr><td>' + html.escape(k) + '</td><td>' + str(v) + '</td>' + ('<td>' + str(extra.get(k, '')) + '</td>' if extra else '') + '</tr>'
    return h + '</tbody></table>'
tiles = [(str(n), 'FINDINGS 条目（M001–M290，M244 已撤回）'), (f'{gpu / n:.1%}', f'在 GPU 上实测复现（{gpu} 条）'), (f'{silent / n:.1%}', f'后果为静默错误（{silent} 条）'),
         (f'{lines:,}', f'行 Lean 代码，{len(L)} 个文件'), (str(thms), f'个定理 / 引理（{defs} 个定义），无 sorry'), (str(lean_link), '条缺陷有对应的 Lean 定理')]
grp = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v[0]}</td><td>{v[1]:,}</td><td>{v[2]}</td></tr>' for k, v in G.items())
body = ''.join(f'''<section class="card"><h2>{html.escape(t)}</h2><div class="chart" id="{i}"></div>
<details><summary>表格视图</summary>{table(r, u, thm_of if i == "c5" else None)}</details></section>''' for i, t, r, u in charts)
page = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MAGMA 审计：缺陷分布与 Lean 建模</title>
<style>
.viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --series-1:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark; --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --series-1:#3987e5; }} }}
:root[data-theme="dark"] .viz-root {{ color-scheme: dark; --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --series-1:#3987e5; }}
body {{ margin:0; }} .viz-root {{ background:var(--surface-1); color:var(--text-primary); font:14px/1.5 -apple-system, "PingFang SC", "Segoe UI", sans-serif; padding:24px; min-height:100vh; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:0 0 12px; }} .sub {{ color:var(--text-secondary); margin:0 0 20px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:24px; }}
.tile {{ border:1px solid var(--grid); border-radius:8px; padding:12px 14px; }} .tile b {{ display:block; font-size:24px; font-variant-numeric:tabular-nums; }} .tile span {{ color:var(--text-secondary); font-size:12px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(460px, 1fr)); gap:16px; }}
.card {{ border:1px solid var(--grid); border-radius:8px; padding:16px; }}
.chart svg {{ display:block; width:100%; height:auto; overflow:visible; }} .chart text {{ font-size:12px; fill:var(--text-secondary); }} .chart .val {{ fill:var(--text-primary); font-variant-numeric:tabular-nums; }}
.chart .hit {{ fill:transparent; cursor:default; }} .chart .row:hover .bar {{ opacity:.8; }}
details {{ margin-top:8px; color:var(--text-secondary); }} table {{ border-collapse:collapse; margin-top:8px; font-size:12px; }} td, th {{ border-bottom:1px solid var(--grid); padding:3px 10px 3px 0; text-align:left; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface-1); color:var(--text-primary); border:1px solid var(--axis); border-radius:6px; padding:6px 9px; font-size:12px; display:none; box-shadow:0 2px 8px rgba(0,0,0,.12); }}
.groups td:nth-child(n+2) {{ font-variant-numeric:tabular-nums; }}
</style></head><body><div class="viz-root">
<h1>MAGMA 2.10.0 审计：缺陷分布与 Lean 建模</h1>
<p class="sub">数据来自 operators/magma/FINDINGS.csv 与 lean/ 下的全部 .lean 文件；由 operators/magma/viz/make_overview.py 生成。</p>
<div class="tiles">{''.join(f'<div class="tile"><b>{v}</b><span>{html.escape(s)}</span></div>' for v, s in tiles)}</div>
<section class="card" style="margin-bottom:16px"><h2>Lean 代码按研究对象分组</h2><table class="groups"><thead><tr><th>对象</th><th>文件</th><th>行数</th><th>定理</th></tr></thead><tbody>{grp}<tr><td><b>合计</b></td><td><b>{len(L)}</b></td><td><b>{lines:,}</b></td><td><b>{thms}</b></td></tr></tbody></table></section>
<div class="grid2">{body}</div><div id="tip"></div>
<script>
const D = {data};
const tip = document.getElementById('tip');
function draw(c) {{
  const el = document.getElementById(c.id), rows = c.rows, max = Math.max(...rows.map(r => r[1]));
  const labelW = 190, valW = 44, rowH = 26, barH = 16, W = 560, plotW = W - labelW - valW, H = rows.length * rowH + 8;
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', rows.map(r => r[0] + ' ' + r[1]).join('，'));
  const base = document.createElementNS(ns, 'line'); base.setAttribute('x1', labelW); base.setAttribute('x2', labelW); base.setAttribute('y1', 0); base.setAttribute('y2', H - 4); base.setAttribute('stroke', 'var(--axis)'); svg.appendChild(base);
  rows.forEach((r, i) => {{
    const y = i * rowH + 4, w = Math.max(2, plotW * r[1] / max), g = document.createElementNS(ns, 'g'); g.setAttribute('class', 'row');
    const t = document.createElementNS(ns, 'text'); t.setAttribute('x', labelW - 8); t.setAttribute('y', y + barH / 2 + 4); t.setAttribute('text-anchor', 'end'); t.textContent = r[0]; g.appendChild(t);
    const b = document.createElementNS(ns, 'path'); const x0 = labelW + 1, x1 = labelW + 1 + w, rr = Math.min(4, w / 2);
    b.setAttribute('d', `M${{x0}},${{y}} H${{x1 - rr}} Q${{x1}},${{y}} ${{x1}},${{y + rr}} V${{y + barH - rr}} Q${{x1}},${{y + barH}} ${{x1 - rr}},${{y + barH}} H${{x0}} Z`);
    b.setAttribute('class', 'bar'); b.setAttribute('fill', 'var(--series-1)'); g.appendChild(b);
    const v = document.createElementNS(ns, 'text'); v.setAttribute('x', x1 + 6); v.setAttribute('y', y + barH / 2 + 4); v.setAttribute('class', 'val'); v.textContent = r[1]; g.appendChild(v);
    const h = document.createElementNS(ns, 'rect'); h.setAttribute('x', 0); h.setAttribute('y', y - 4); h.setAttribute('width', W); h.setAttribute('height', rowH); h.setAttribute('class', 'hit'); g.appendChild(h);
    const total = rows.reduce((s, q) => s + q[1], 0), extra = c.id === 'c5' ? `，${{D.thm[r[0]]}} 个定理` : `，占 ${{(100 * r[1] / total).toFixed(1)}}%`;
    h.addEventListener('mousemove', e => {{ tip.style.display = 'block'; tip.textContent = `${{r[0]}}：${{r[1]}} ${{c.unit}}${{extra}}`; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }});
    h.addEventListener('mouseleave', () => tip.style.display = 'none');
    svg.appendChild(g);
  }});
  el.appendChild(svg);
}}
D.charts.forEach(draw);
</script></div></body></html>'''
open(os.path.join(ROOT, 'overview.html'), 'w').write(page)
print('wrote', os.path.join(ROOT, 'overview.html'), n, gpu, silent, lines, thms, defs, lean_link, dict(G))
