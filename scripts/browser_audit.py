"""OPTIONAL real-browser audit (Chromium via Playwright + axe-core, WCAG 2.0/2.1/2.2 A+AA rules).
Not part of the test suite and NOT run by pytest. Needs:  pip install playwright && playwright install chromium ;  npm install axe-core
Run against the QA harness (synthetic inputs):   python tests/qa_harness.py &   AXE_JS=node_modules/axe-core/axe.min.js python scripts/browser_audit.py
Reports console/page errors, horizontal overflow, keyboard focus visibility and axe violations for every route at 3 viewports.
axe finds only a subset of accessibility problems; a clean run is not a conformance claim."""
import json, sys, time
from playwright.sync_api import sync_playwright
import os
BASE=os.environ.get("AUDIT_BASE","http://127.0.0.1:5057")
AXE=open(os.environ.get("AXE_JS","node_modules/axe-core/axe.min.js")).read()   # npm install axe-core
ROUTES=["overview","research","forecast","evaluation","backtest","ai","portfolio","risk","health","settings"]
VPS={"desktop":(1440,900),"tablet":(820,1100),"mobile":(390,844)}
problems=[]; axe_all={}; overflow={}; focus_report={}; artefacts=[]
import re
BAD=re.compile(r"\[object |\bundefined\b|\bNaN\b|\bnull\b")
def flow(pg, name, on_route):
    pg.goto(BASE+"/"); pg.wait_for_selector("#view h1")
    pg.fill("#sym","AAPL"); pg.click("#symform button"); pg.wait_for_selector("#quote .badge")
    pg.click("#symform button"); time.sleep(1.2)            # second load is served from the server's quote cache -> "Cached" badge
    if "[object" in pg.inner_text("#quote") or pg.locator("#quote .badge").count() < 1: artefacts.append((name,"quote-strip",pg.inner_text("#quote")[:80]))
    pg.click("a[href='#/forecast']"); pg.wait_for_selector("#run"); pg.fill("#fe","2"); pg.click("#run")
    pg.wait_for_selector("text=Numerical ensemble", timeout=420000)
    for r in ROUTES:
        pg.click(f"a[href='#/{r}']"); time.sleep(0.9)
        if r=="backtest": pg.click("button:has-text('Run backtest')"); pg.wait_for_selector("text=Trade ledger"); time.sleep(0.5)
        if r=="ai": pg.click("#gen"); pg.wait_for_selector("text=AI-generated", timeout=30000); time.sleep(0.3)
        on_route(pg, r)
with sync_playwright() as p:
    b=p.chromium.launch()
    for name,(w,h) in VPS.items():
        # --- context A: CSP enforced -> console/page errors, overflow, keyboard focus
        ctx=b.new_context(viewport={"width":w,"height":h}); pg=ctx.new_page()
        pg.on("console", lambda m,n=name: problems.append((n,m.type,m.text[:150])) if m.type in("error","warning") else None)
        pg.on("pageerror", lambda e,n=name: problems.append((n,"pageerror",str(e)[:150])))
        pg.on("requestfailed", lambda r,n=name: problems.append((n,"requestfailed",r.url)))
        def a_route(pg,r,n=name):
            overflow[(n,r)]=pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            for m_ in BAD.finditer(pg.inner_text("#view") + " " + pg.inner_text("#quote")):
                txt=pg.inner_text("#view"); i_=txt.find(m_.group(0)); artefacts.append((n,r,txt[max(0,i_-40):i_+40].replace("\n"," ")))
            if r in("overview","forecast","settings"):
                seen=[]
                pg.evaluate("document.activeElement && document.activeElement.blur()")
                for _ in range(12):
                    pg.keyboard.press("Tab")
                    info=pg.evaluate("""(()=>{const e=document.activeElement; if(!e||e===document.body) return null; const c=getComputedStyle(e);
                      return {tag:e.tagName, name:(e.getAttribute('aria-label')||e.innerText||e.id||'').slice(0,25), outline:c.outlineStyle+' '+c.outlineWidth, ow:parseFloat(c.outlineWidth)}})()""")
                    if info: seen.append(info)
                focus_report[(n,r)]=(len(seen), sum(1 for s in seen if s["outline"].startswith("solid") and s["ow"]>=2))
        flow(pg,name,a_route); ctx.close()
        # --- context B: CSP bypassed only to inject axe-core
        ctx=b.new_context(viewport={"width":w,"height":h}, bypass_csp=True); pg=ctx.new_page()
        def b_route(pg,r,n=name):
            pg.add_script_tag(content=AXE)
            res=pg.evaluate("""async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}})""")
            axe_all[(n,r)]=[(v["id"],v["impact"],len(v["nodes"]),v["nodes"][0]["target"][0][:60] if v["nodes"] else "", v["nodes"][0].get("failureSummary","")[:140].replace("\n"," ")) for v in res["violations"]]
        flow(pg,name,b_route); ctx.close()
    b.close()
print("== console/page/request problems:", problems or "none")
print("== rendering artefacts ([object / undefined / NaN / null) in visible text:", sorted(set(artefacts)) or "none")
print("== horizontal overflow px (nonzero only):", {f"{k[0]}:{k[1]}":v for k,v in overflow.items() if v} or "none on any route/viewport")
print("== keyboard focus (tabs reached, with >=2px solid outline):", {f"{k[0]}:{k[1]}":v for k,v in focus_report.items()})
n=sum(len(v) for v in axe_all.values()); print(f"== axe WCAG 2.2 AA violations: {n} across {len(axe_all)} route/viewport pages")
seen=set()
for k,v in axe_all.items():
    for item in v:
        key=(item[0],item[3])
        if key in seen: continue
        seen.add(key); print(f"  [{k[0]}:{k[1]}] {item[0]} ({item[1]}) x{item[2]} e.g. {item[3]} :: {item[4]}")
