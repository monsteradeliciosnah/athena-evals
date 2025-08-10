from __future__ import annotations
import json, asyncio, pathlib, statistics as stats
import typer
from pydantic import BaseModel
from .metrics import exact_match, f1_score, rouge_l
from .models import LocalEcho, OpenAIClient
from .report import write_report

app = typer.Typer()

class Row(BaseModel):
    prompt: str
    gold: str
    meta: dict | None = None

def load_rows(path: str):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip(): continue
        rows.append(Row(**json.loads(line)))
    return rows

async def run_eval(dataset: str, model: str, report: str):
    rows = load_rows(dataset)
    client = LocalEcho() if model=="local-echo" else OpenAIClient(model=model)
    records = []
    ems, f1s, rls = [], [], []
    for r in rows:
        gen = await client.generate(r.prompt)
        pred = gen["output"]
        em = exact_match(pred, r.gold); f1 = f1_score(pred, r.gold); rl = rouge_l(pred, r.gold)
        ems.append(em); f1s.append(f1); rls.append(rl)
        records.append({"prompt": r.prompt, "pred": pred, "gold": r.gold, "em": em, "f1": f1})
    write_report(report, records, stats.mean(ems), stats.mean(f1s), stats.mean(rls), model)
    typer.echo(f"Wrote report to {report}")

@app.command()
def eval(dataset: str = "datasets/demo.jsonl", model: str = "local-echo", report: str = "reports/report.html"):
    pathlib.Path("reports").mkdir(parents=True, exist_ok=True)
    asyncio.run(run_eval(dataset, model, report))

if __name__ == "__main__":
    app()
