"""Report the preregistered paired protocol experiment without selecting winners."""
import argparse
import json
import statistics
from pathlib import Path
from experiments.report_deepseek import bootstrap_delta

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/deepseek-structured-v2')
    parser.add_argument('--output', default='artifacts/deepseek_structured')
    args=parser.parse_args()
    source=Path(args.source)
    out=Path(args.output)
    data=json.loads((source/'results.json').read_text(encoding='utf-8'))
    cases=json.loads((source/'dataset.json').read_text(encoding='utf-8'))['test']
    assert data['status']=='complete'
    assert len(data['records'])==54 and len(data['provider_observations'])==54
    measured={(r['case_id'],r['route'],r['protocol']):r for r in data['records'] if r['phase']=='test'}
    assert len(measured)==36
    strategies={}
    for route in ('budget','strong','rsl'):
        for protocol in ('plain','structured'):
            strategies[route+'_'+protocol]=[measured[c['id'],data['frozen_routes'][c['id']]['selected_model'] if route=='rsl' else route,protocol] for c in cases]
    summary={name:dict(correct=int(sum(r['score'] for r in rows)),count=len(rows),accuracy=statistics.mean(r['score'] for r in rows),cost_upper_cny=sum(r['cost_upper_cny'] for r in rows),avg_latency_ms=statistics.mean(r['provider_latency_ms'] for r in rows),truncated=sum(r['finish_reason']=='length' for r in rows)) for name,rows in strategies.items()}
    interval=bootstrap_delta(strategies['rsl_structured'],strategies['rsl_plain'])
    report=dict(strategies=summary,paired_accuracy_delta=summary['rsl_structured']['accuracy']-summary['rsl_plain']['accuracy'],exploratory_95_percent_bootstrap=interval,all_calls_cost_upper_cny=sum(r['cost_upper_cny'] for r in data['records']),budget_reserved_total_cny=data['budget_reserved_cny_total'])
    out.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','dataset.json','results.json'):
        (out/name).write_bytes((source/name).read_bytes())
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# 结构化输出改进：真实 API 配对实验','',
      '使用新的训练种子 7023、测试种子 7024；18 道训练题、9 道独立测试题（三类各 3 道）。共 54 次真实调用。',
      '训练只使用结构化输出，反馈仍由程序计算。测试前冻结 RSL 选择，每题分别调用两模型的普通和结构化输出；顺序交替。测试不回写反馈。',
      '主比较为同一冻结 RSL 路由在两种输出协议下的差异。此实验隔离的是输出协议，不是 RSL 算法本身的增益，也不是与第一轮不同测试集直接比较。',
      '结构化协议使用 response_format=json_object 及不含正确答案的提示词；严格检查单一 answer 字段及类型，然后转换回整数/数组并使用原评分器。错误答案、额外字段、错误类型均不通过。','',
      '|策略|通过数|成功率|费用上界/元|平均模型耗时/ms|截断数|','|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f"|{name}|{s['correct']}/{s['count']}|{s['accuracy']:.1%}|{s['cost_upper_cny']:.6f}|{s['avg_latency_ms']:.0f}|{s['truncated']}|")
    lines += ['',f"RSL 结构化输出相对普通输出成功率变化：{report['paired_accuracy_delta']:+.1%}；探索性配对 bootstrap 95% 区间 {interval}。",f"本轮全部调用费用保守估算：{report['all_calls_cost_upper_cny']:.6f} 元；累计预算预留 {data['budget_reserved_cny_total']:.2f} 元（不是实际账单）。",'',
      '样本只有 9 题，不能据此声称统计显著、普遍有效或超过最佳静态模型。耗时受网络和缓存影响。原失败实验完整保留，未根据本轮测试结果重新调参或挑选运行。',
      '', '复现：先配置 .env，然后 python -m experiments.run_structured --execute；每次需要新的输出目录，且共享预算账本必须有足够余额。报告：python -m experiments.report_structured。',
      '价格按 2026-10-23 峰时缓存未命中估算；JSON 模式文档：https://api-docs.deepseek.com/guides/json_mode/']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
