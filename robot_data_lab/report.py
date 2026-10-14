from __future__ import annotations

import html
from pathlib import Path

from .io import read_json


def render_report(audit, summary, provenance, destination):
    escape = html.escape
    rows = []
    for ep in audit["episodes"]:
        status = "无效" if not ep["valid"] else ("保留" if ep["selected"] else "待复核")
        reasons = ", ".join(ep["errors"] + ep["warnings"]) or "通过当前规则"
        idle = "—" if ep["idle_fraction"] is None else f"{ep['idle_fraction']:.1%}"
        rows.append(f'<tr><td>{ep["episode_index"]:03d}</td><td>{ep["frames"]}</td><td>{status}</td><td>{idle}</td>'
                    f'<td>{ep["video_samples_checked"]}</td><td>{escape(reasons)}</td></tr>')
    experiment_rows = []
    for r in summary["results"]:
        experiment_rows.append(f'<tr><td>{escape(r["arm"])}</td><td>{r["seed"]}</td><td>{r["train_episodes"]}</td>'
            f'<td>{r["mae_native_units"]:.4f}</td><td>{r["normalized_mae"]:.4f}</td><td>{r["episode_macro_mae"]:.4f}</td>'
            f'<td>{r["validation_mean_action_mae"]:.4f}</td><td>{r.get("validation_copy_state_mae",0):.4f}</td><td>{r["seconds"]:.1f}s</td></tr>')
    cards = ''.join(f'<article><small>{label}</small><strong>{value}</strong></article>' for label, value in [
        ("真实演示", audit["total_episodes"]), ("原始帧", f'{audit["total_frames"]:,}'),
        ("结构有效", audit["valid_episodes"]), ("规则保留", audit["selected_episodes"])])
    charts = []
    max_mae = max((v["mean_mae"] for v in summary["aggregate"].values()), default=1) or 1
    for name, val in summary["aggregate"].items():
        charts.append(f'<div class="barrow"><span>{escape(name)}</span><div class="track"><div class="bar" style="width:{100*val["mean_mae"]/max_mae:.2f}%"></div></div><b>{val["mean_mae"]:.4f} ± {val["std_mae"]:.4f}</b></div>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Data Lab · 实验报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:#193037;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1220px;margin:auto;padding:46px 30px}header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}h1{font-size:38px;letter-spacing:-1.5px;margin:6px 0}h2{font-size:21px;margin:0 0 16px}p{color:#55686f}small,.eyebrow{color:#53767a;font-size:12px;text-transform:uppercase;letter-spacing:1.5px}.badge{background:#dbefe7;color:#166849;border-radius:8px;padding:8px 14px;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:26px 0}article,section{background:white;border:1px solid #e0e8e9;border-radius:14px;padding:22px}article strong{display:block;font-size:34px;margin-top:8px}section{margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px 10px;text-align:left;border-bottom:1px solid #e8eff0}th{color:#5e7277;white-space:nowrap}td:last-child{overflow-wrap:anywhere}.scroll{overflow-x:auto}.notice{background:#fff8e8;border-left:4px solid #e0a341;padding:16px 20px;border-radius:8px;color:#71551e}input{padding:11px 14px;border:1px solid #cbdadc;border-radius:7px;width:min(420px,100%);margin:0 0 16px}.barrow{display:flex;align-items:center;gap:16px;margin:16px 0}.barrow span{width:120px}.barrow b{font-size:13px;min-width:145px}.track{background:#eef3f4;height:16px;flex:1;border-radius:4px}.bar{height:100%;background:#26847b;border-radius:4px}code{font-size:12px;overflow-wrap:anywhere}a{color:#20796e}.status{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.status div{padding:12px;background:#f2f6f6;border-radius:8px}.status b{display:block}.pending{color:#986c27}footer{font-size:12px;color:#75888f;margin:25px 0}@media(max-width:700px){main{padding:24px 14px}h1{font-size:28px}.cards,.status{grid-template-columns:repeat(2,1fr)}header{display:block}.badge{display:inline-block}.barrow{gap:8px}.barrow b{min-width:100px}}</style>
<main><header><div><div class="eyebrow">ROBOT DATA LAB / EXPERIMENT RECORD</div><h1>从真实轨迹到可复核的实验</h1><p>LeRobot 数据质检 · 可追溯 Pipeline · CPU 模仿学习基线</p></div><span class="badge">本地实测报告</span></header>
CARDS
<div class="notice"><b>结果边界：</b>本页指标来自真实公开数据上的离线验证。CPU 基线是 RGB 网格特征 + 关节状态的 MLP，不是 VLA。离线动作误差不等于机器人任务成功率。SmolVLA 训练、LIBERO 闭环评测和真机实验尚未执行。</div>
<section><h2>执行状态</h2><div class="status"><div><b>数据质检</b>已执行</div><div><b>CPU 基线</b>已执行</div><div class="pending"><b>SmolVLA</b>待 Linux / GPU 运行</div><div class="pending"><b>LIBERO</b>待闭环评测</div></div></section>
<section><h2>离线验证结果</h2><p>按完整 episode 划分训练与验证，归一化仅拟合训练集。MAE 使用数据原生动作单位；混合关节平均值仅用于本数据集内比较。条形图为跨种子均值 ± 样本标准差。</p>CHARTS<p>COMPARISON</p><div class="scroll"><table><thead><tr><th>实验组</th><th>种子</th><th>训练轨迹</th><th>MAE ↓</th><th>归一化 MAE ↓</th><th>轨迹宏平均 MAE ↓</th><th>均值预测 MAE</th><th>复制状态 MAE</th><th>训练耗时</th></tr></thead><tbody>EXPERIMENTS</tbody></table></div><p>“复制状态”是当前关节状态直接作为绝对目标动作的强基线，可暴露低离线误差的局限。对照过滤实验使用同一验证集和训练步数；没有筛掉训练轨迹时跳过重复实验。</p></section>
<section><h2>逐轨迹数据审计</h2><p>视频采用抽样解码检查；低纹理、停顿等告警需要人工复核，不能自动解释为任务失败。</p><input id="search" placeholder="筛选轨迹编号 / 告警原因 / 状态" aria-label="筛选轨迹"><div class="scroll"><table id="episodes"><thead><tr><th>Episode</th><th>帧数</th><th>筛选</th><th>停顿比例</th><th>已检视频帧</th><th>检查结果</th></tr></thead><tbody>ROWS</tbody></table></div></section>
<section><h2>可追溯信息</h2><p>数据集：<a href="https://huggingface.co/datasets/REPO">REPO</a><br>数据版本：<code>REVISION</code><br>代码指纹：<code>CODEHASH</code><br>配置指纹：<code>CFGHASH</code><br>验证轨迹：VALIDATION</p><p>完整机器可读结果：<a href="audit.json">audit.json</a> · <a href="experiments.json">experiments.json</a> · <a href="provenance.json">provenance.json</a></p></section>
<footer>Robot Data Lab · Research / portfolio prototype · 自动标注与真机采集未声称完成。</footer></main><script>document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#episodes tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'''
    replacements = {"CARDS": '<div class="cards">'+cards+'</div>', "CHARTS": ''.join(charts), "ROWS": ''.join(rows),
        "EXPERIMENTS": ''.join(experiment_rows), "COMPARISON": escape(summary["comparison_note"]),
        "REPO": escape(audit["dataset"]["repo_id"]), "REVISION": escape(audit["dataset"]["revision"]),
        "CODEHASH": escape(provenance["code_fingerprint"]), "CFGHASH": escape(provenance["config_fingerprint"]),
        "VALIDATION": escape(str(summary["split"]["validation"]))}
    for key, value in replacements.items():
        page = page.replace(key, value)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
