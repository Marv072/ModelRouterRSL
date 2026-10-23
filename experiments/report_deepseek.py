"""Create shareable, credential-free artifacts from the real experiment."""
import argparse
import csv
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path


def bootstrap_delta(left, right):
    rng = random.Random(17)
    differences = [a['score']-b['score'] for a,b in zip(left,right)]
    samples = sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
    return [samples[49],samples[1949]]


def build(source, output):
    source, output = Path(source), Path(output)
    data = json.loads((source/'results.json').read_text(encoding='utf-8'))
    dataset = json.loads((source/'dataset.json').read_text(encoding='utf-8'))
    if data['status'] != 'complete':
        raise ValueError('Incomplete run: do not present partial results as completed')
    train = [r for r in data['records'] if r['phase']=='train']
    measured = {(r['case_id'],r['route']):r for r in data['records'] if r['phase']=='test'}
    assert len(train)==48 and len(measured)==48 and len(data['provider_observations'])==96
    strategies = {'always_flash':[], 'always_pro':[], 'rsl_frozen':[]}
    for case in dataset['test']:
        cid = case['id']
        strategies['always_flash'].append(measured[cid,'budget'])
        strategies['always_pro'].append(measured[cid,'strong'])
        strategies['rsl_frozen'].append(measured[cid,data['frozen_routes'][cid]['selected_model']])
    policy = data['manifest']['policy']

    def summarize(rows):
        costs = [r['cost_upper_cny'] for r in rows]
        if any(c is None for c in costs):
            raise ValueError('Unknown usage costs; cannot claim a complete cost estimate')
        utility = [policy['rsl_quality_weight']*r['score']
                   -policy['rsl_cost_weight']*min(r['cost_upper_cny']/policy['rsl_cost_scale'],1)
                   -policy['rsl_latency_weight']*min(r['provider_latency_ms']/policy['rsl_latency_scale_ms'],1)
                   for r in rows]
        return dict(count=len(rows),correct=int(sum(r['score'] for r in rows)),
            accuracy=statistics.mean(r['score'] for r in rows), cost_upper_cny=sum(costs),
            avg_provider_latency_ms=statistics.mean(r['provider_latency_ms'] for r in rows),
            utility=statistics.mean(utility), selected=dict(Counter(r['route'] for r in rows)))

    summary = {name:summarize(rows) for name,rows in strategies.items()}
    categories = {category:{name:summarize([r for r in rows if r['category']==category])
                  for name,rows in strategies.items()} for category in ('math','coding','general')}
    intervals = {name:bootstrap_delta(strategies['rsl_frozen'],strategies[name])
                 for name in ('always_flash','always_pro')}
    report = dict(summary=summary,by_category=categories,training=summarize(train),
        all_paid_calls_cost_upper_cny=sum(r['cost_upper_cny'] for r in data['records']),
        all_paid_calls=96,accuracy_delta_95_percentile_bootstrap=intervals,
        reservation_including_prior_probe_cny=data['budget_reserved_cny_total'])
    diagnostics = {}
    for route in ('budget','strong'):
        rows = [r for r in measured.values() if r['route']==route]
        diagnostics[route] = dict(truncated=sum(r['finish_reason']=='length' for r in rows),
            numeric_format_violation=sum(r['category'] in ('math','coding') and
                re.fullmatch(r'[+-]?\d+',r['answer'].strip()) is None for r in rows),
            wrong_bare_integer=sum(r['category'] in ('math','coding') and
                re.fullmatch(r'[+-]?\d+',r['answer'].strip()) is not None and r['score']==0 for r in rows))
    report['test_diagnostics'] = diagnostics
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # These run files contain generated public questions, answers and statistics only.
    # The credential-bearing DBs, environment, headers and budget ledger are excluded.
    for name in ('results.json','manifest.json','dataset.json'):
        content = (source/name).read_text(encoding='utf-8')
        (output/name).write_text(content,encoding='utf-8')
    with (output/'learning_curve.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=['round','correct','accuracy','cost_upper_cny','budget_count','strong_count'])
        writer.writeheader()
        for end in (12,24,36,48):
            part = summarize(train[end-12:end])
            writer.writerow(dict(round=end,correct=part['correct'],accuracy=part['accuracy'],
                cost_upper_cny=part['cost_upper_cny'],budget_count=part['selected'].get('budget',0),
                strong_count=part['selected'].get('strong',0)))
    lines = ['# DeepSeek 真实 RSL 实验报告','',
        f"执行时间（UTC）：{data['started_utc']} 至 {data['completed_utc']}。",'',
        '**本报告使用真实 DeepSeek API 返回结果，不是模拟数据。**','',
        '## 实验设计','',
        '- 候选：deepseek-flash（budget）与 deepseek-v4-pro（strong）；关闭思考，temperature=0，最多 256 输出 token。',
        '- 48 道在线学习题，三类各 16 道；24 道独立测试题，三类各 8 道。种子、题目和策略在调用前固定。',
        '- 数学为乘加取模；代码为短循环执行结果；general 为筛选排序后的 JSON ID 数组。标签由程序计算。',
        '- 训练通过实际 /v1/chat/completions auto 路由，再通过反馈接口提交客观分数；使用新的隔离数据库。',
        '- 测试前冻结所有选择，两模型分别回答全部测试题，交替调用顺序。RSL 从实测结果中取被选模型的答案。',
        '- 这是冻结策略配对回放，不是额外发送第三组 RSL 请求。三种策略共享相同的模型输出，测试不回写学习数据。',
        '- 无 Judge、无自动重试；共 96 次真实请求。输入与输出有限制，每次调用前预留 0.03 元。','',
        '## 冻结测试结果','',
        '| 策略 | 严格通过数 | 任务成功率（含格式） | 24 题费用上界估算/元 | 平均模型耗时/ms | 综合效用 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in summary.items():
        lines.append(f"| {name} | {row['correct']}/24 | {row['accuracy']:.2%} | {row['cost_upper_cny']:.6f} | {row['avg_provider_latency_ms']:.0f} | {row['utility']:.4f} |")
    lines += ['','费用统一采用官方高峰缓存未命中单价：Flash 输入 2、输出 8 元/百万 token，Pro 输入 9、输出 27 元/百万 token。缓存和空闲时段可能使实际费用更低；不是账单扣款数。',
        '时延为本次模型调用耗时，不含冻结策略预览及反馈写入；小样本、顺序、缓存与网络会影响结果。',
        '综合效用使用预先固定的质量权重 1、成本权重 0.2、延迟权重 0.1，费用尺度 0.005 元、延迟尺度 5000ms，两项惩罚均截断为 1。','',
        '## 分类结果与选择','',
        '| 类别 | Flash 正确数 | Pro 正确数 | RSL 正确数 | RSL 选择 |',
        '|---|---:|---:|---:|---|']
    for cat,rows in categories.items():
        lines.append(f"| {cat} | {rows['always_flash']['correct']}/8 | {rows['always_pro']['correct']}/8 | {rows['rsl_frozen']['correct']}/8 | {json.dumps(rows['rsl_frozen']['selected'])} |")
    lines += ['','## 失败分析','',
        '主指标要求回答数值/JSON 正确且符合“只返回答案”的格式，不能把它解读为不受格式限制的推理正确率。',
        '| 模型 | 输出被截断 | 数字题格式不合规 | 返回纯整数但算错 |',
        '|---|---:|---:|---:|']
    for route,row in diagnostics.items():
        lines.append(f"| {route} | {row['truncated']} | {row['numeric_format_violation']} | {row['wrong_bare_integer']} |")
    lines += ['', '这些诊断计数可重叠。256 token 输出上限与非思考模式是预先固定的条件；长解释、截断或格式不符都可能计 0 分。主评分未在看见结果后放宽。']
    for route in ('budget','strong'):
        general_train = [r for r in train if r['category']=='general' and r['route']==route]
        lines.append(f"- general 类训练中 {route} 通过 {int(sum(r['score'] for r in general_train))}/{len(general_train)}；独立测试通过 {categories['general']['always_flash' if route=='budget' else 'always_pro']['correct']}/8。")
    lines.append('少量历史样本的排序不能保证在后续题目中保持。下面的配对结果与区间用于说明本次得失，不能据此宣称普遍优于静态路由。')
    lines += ['','## 预算与训练开销','',
        f"- 48 次训练调用费用上界估算：{report['training']['cost_upper_cny']:.6f} 元。",
        f"- 本轮全部 96 次调用（含两组测试）的费用上界估算：{report['all_paid_calls_cost_upper_cny']:.6f} 元。",
        f"- 连同先前连通性测试，累计保守预留 {report['reservation_including_prior_probe_cny']:.2f} 元，小于 5 元；预留不等于扣费。",
        '- 不能只比较测试成本就宣称整体省钱；实际部署还应计入训练/探索开销，并评估需要多少后续请求才能摊平。','',
        '## 配对不确定性与结论边界','']
    for name,interval in intervals.items():
        delta = summary['rsl_frozen']['accuracy']-summary[name]['accuracy']
        lines.append(f"- RSL 相对 {name} 的准确率差：{delta:+.2%}；配对 bootstrap 95% 百分位区间 [{interval[0]:+.2%}, {interval[1]:+.2%}]（2000 次、固定种子 17）。")
    lines += ['','仅 24 道测试、一次实际运行，区间只反映此任务样本的抽样波动，不包含模型重复调用随机性，也不是泛化提升证明。未根据测试结果修改策略权重或重跑挑选最好结果。',
        '本实验验证真实接口上的反馈闭环及特定任务集的策略权衡。任务是程序生成的三个模板族，训练/测试只有数值与记录不同，不能外推到开放问答、自然写作或真实复杂编码任务。',
        '完整问题、参考答案、模型回答、用量、决策和学习轨迹见同目录 JSON/CSV。参考答案从未发送给模型；没有使用模型生成代码执行。','',
        '## 复现','',
        '```bash','python -m experiments.run_deepseek  # 只显示计划，不调用模型',
        'python -m experiments.run_deepseek --execute --output runs/deepseek-new-run',
        'python -m experiments.report_deepseek --source runs/deepseek-new-run --output artifacts/deepseek_new_run','```','',
        '执行需有效 .env，并消耗余额；脚本保留已有累计预算，剩余额度不足时拒绝运行。不要删除预算账本以绕过上限。',
        '官方价格：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='runs/deepseek-real-v1')
    parser.add_argument('--output',default='artifacts/deepseek_real')
    args = parser.parse_args()
    build(args.source,args.output)
