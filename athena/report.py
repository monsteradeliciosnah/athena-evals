from __future__ import annotations

from pathlib import Path

from jinja2 import Template

TEMPLATE = Template("""
<!doctype html>
<html><head><meta charset="utf-8"><title>Athena Evals</title>
<style>body{font-family:system-ui;margin:20px} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:8px}</style>
</head><body>
<h1>Athena Evaluation Report</h1>
<p>Model: {{ model_name }} · Samples: {{ n }} · EM: {{ em|round(3) }} · F1: {{ f1|round(3) }} · RougeL: {{ rl|round(3) }}</p>
<table><thead><tr><th>#</th><th>Prompt</th><th>Prediction</th><th>Gold</th><th>EM</th><th>F1</th></tr></thead>
<tbody>
{% for r in rows %}
<tr><td>{{ loop.index }}</td><td>{{ r.prompt }}</td><td>{{ r.pred }}</td><td>{{ r.gold }}</td><td>{{ '%.2f'|format(r.em) }}</td><td>{{ '%.2f'|format(r.f1) }}</td></tr>
{% endfor %}
</tbody></table>
</body></html>
""")


def write_report(path: str, rows, em, f1, rl, model_name: str):
    html = TEMPLATE.render(
        rows=rows, em=em, f1=f1, rl=rl, n=len(rows), model_name=model_name
    )
    Path(path).write_text(html, encoding="utf-8")
    return path
