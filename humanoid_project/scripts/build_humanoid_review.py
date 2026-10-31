"""Build the synchronized humanoid video, torque, speed, and power dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = ("stand", "walk", "sprint", "getup")
LABELS = {"stand": "Standing", "walk": "Walking at 1 m/s", "sprint": "4 m/s sprint",
          "getup": "Get-up refinement"}


def load_capture(root: Path, skill: str) -> dict:
    folder = root / skill
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    joints = manifest["joints"]
    rows = []
    with (folder / manifest["csv"]).open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row = [round(float(source[k]), 4) for k in
                   ("time_s", "episode_time_s", "base_height_m", "command_vx_m_s", "base_vx_m_s")]
            row += [round(float(source[f"{j['slug']}_applied_torque_nm"]), 4) for j in joints]
            row += [round(float(source[f"{j['slug']}_velocity_rad_s"]), 4) for j in joints]
            row += [round(float(source[f"{j['slug']}_mechanical_power_w"]), 4) for j in joints]
            rows.append(row)
    moving = [r[4] for r in rows if r[0] >= min(2.0, manifest["duration_s"] * 0.2)]
    mean_speed = sum(moving) / len(moving) if moving else 0.0
    return {"label": LABELS[skill], "video": f"{skill}/{manifest['video']}",
            "csv": f"{skill}/{manifest['csv']}", "duration": manifest["duration_s"],
            "checkpoint": manifest["checkpoint"], "joints": joints, "rows": rows,
            "meanSpeed": round(mean_speed, 3)}


def build_html(captures: dict) -> str:
    encoded = json.dumps(captures, separators=(",", ":"))
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Five-foot humanoid training review</title><style>
:root{color-scheme:dark;--bg:#0b0f14;--panel:#141b24;--panel2:#1b2531;--line:#344252;--text:#edf4fb;--muted:#9cadbf;--accent:#62c4ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}main{max-width:1720px;margin:auto;padding:18px}h1{font-size:23px;margin:0 0 5px}.subtitle,.note{color:var(--muted);font-size:13px;line-height:1.45}.skills{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}button{font:600 14px Segoe UI;border:1px solid var(--line);background:var(--panel);color:var(--text);padding:8px 15px;cursor:pointer}button.active{background:#174a67;border-color:var(--accent)}.layout{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(600px,1fr);gap:16px;align-items:start}.panel,.chart{background:var(--panel);border:1px solid var(--line);padding:14px}video{display:block;width:100%;max-height:650px;background:#000}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px}.metric{background:var(--panel2);padding:9px}.metric span{display:block;color:var(--muted);font-size:11px}.metric strong{font:600 17px Consolas}.table-wrap{max-height:330px;overflow:auto}table{width:100%;border-collapse:collapse;font:12px Consolas}th,td{padding:6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{position:sticky;top:0;background:var(--panel)}thead th{font:11px Segoe UI;color:var(--muted)}.charts{display:grid;gap:14px;margin-top:16px}.chart h2{font-size:15px;margin:0 0 8px}.chart canvas{display:block;width:100%;height:330px;background:#0e141b}.legend{display:flex;flex-wrap:wrap;gap:6px 15px;margin-top:8px;font-size:11px;color:var(--muted)}.legend span:before{content:'';display:inline-block;width:15px;height:3px;margin:0 5px 2px 0;background:var(--c)}a{color:var(--accent)}@media(max-width:1100px){.layout{grid-template-columns:1fr}.table-wrap{max-height:none}}@media(max-width:620px){main{padding:10px}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main><h1>Five-foot humanoid — synchronized training review</h1><div class="subtitle">Recorded policy motion synchronized with applied joint torque, joint speed, and mechanical power. “Actual” speed is the robot’s body-frame forward velocity.</div><nav class="skills"></nav><div class="layout"><section class="panel"><video id="video" controls preload="metadata"></video><div class="note" id="source"></div></section><section class="panel"><div class="metrics"><div class="metric"><span>Simulation time</span><strong id="time">0.00 s</strong></div><div class="metric"><span>Command / actual Vx</span><strong id="velocity">0 / 0</strong></div><div class="metric"><span>Base height</span><strong id="height">0 m</strong></div><div class="metric"><span>Mean actual Vx</span><strong id="mean">0 m/s</strong></div></div><div class="table-wrap"><table><thead><tr><th>Joint</th><th>Applied τ</th><th>Isaac limit</th><th>Use</th><th>ω</th><th>P=τ·ω</th></tr></thead><tbody id="joint-rows"></tbody></table></div></section></div><div class="charts"><section class="chart"><h2>Applied joint torque over time [N·m]</h2><canvas id="torque"></canvas><div class="legend" id="torque-legend"></div></section><section class="chart"><h2>Mechanical joint power over time [W]</h2><canvas id="power"></canvas><div class="legend" id="power-legend"></div></section></div></main><script>
const captures=__CAPTURES__,order=['stand','walk','sprint','getup'],colors=['#5bc0eb','#3388dd','#9bc53d','#64b547','#fde74c','#f5a623','#fa7921','#e94f37','#d36ac2','#9b5de5','#7d8cc4','#b8c0ff','#00d4aa','#ff6b9d','#63e6be','#ffd43b','#748ffc','#ff922b','#51cf66','#cc5de8','#74c0fc','#ffa94d'];const video=document.getElementById('video');let data=captures.stand;const nav=document.querySelector('.skills');for(const key of order){const b=document.createElement('button');b.textContent=captures[key].label;b.dataset.skill=key;b.onclick=()=>selectSkill(key);nav.appendChild(b)}function label(s){return s.replaceAll('_',' ').replace(/\\b\\w/g,c=>c.toUpperCase())}function nearest(t){let a=data.rows,lo=0,hi=a.length-1;while(lo<hi){let m=(lo+hi)>>1;if(a[m][0]<t)lo=m+1;else hi=m}if(lo&&Math.abs(a[lo-1][0]-t)<Math.abs(a[lo][0]-t))lo--;return a[lo]}function selectSkill(k){data=captures[k];document.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.skill===k));video.src=data.video;video.load();document.getElementById('source').innerHTML=data.label+' · '+data.duration.toFixed(0)+' s · <a href="'+data.csv+'">telemetry CSV</a>';document.getElementById('mean').textContent=data.meanSpeed.toFixed(2)+' m/s';makeTable();makeLegends();draw()}function makeTable(){let body=document.getElementById('joint-rows');body.innerHTML='';data.joints.forEach((j,i)=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+label(j.name)+'</td><td id="t'+i+'">--</td><td>'+j.effort_limit_nm.toFixed(1)+' N·m</td><td id="u'+i+'">--</td><td id="w'+i+'">--</td><td id="p'+i+'">--</td>';body.appendChild(tr)})}function makeLegends(){for(const id of ['torque-legend','power-legend']){let e=document.getElementById(id);e.innerHTML='';data.joints.forEach((j,i)=>{let s=document.createElement('span');s.style.setProperty('--c',colors[i%colors.length]);s.textContent=label(j.name);e.appendChild(s)})}}function fit(c){let d=Math.min(devicePixelRatio||1,2),r=c.getBoundingClientRect(),w=Math.max(600,Math.round(r.width*d)),h=Math.round(330*d);if(c.width!==w||c.height!==h){c.width=w;c.height=h}return{w,h,d}}function plot(id,off,unit,t){let c=document.getElementById(id),z=fit(c),x=c.getContext('2d'),w=z.w,h=z.h,d=z.d,p={l:68*d,r:18*d,t:25*d,b:42*d},n=data.joints.length,tmax=data.rows.at(-1)[0],ymax=1;for(const row of data.rows)for(let j=0;j<n;j++)ymax=Math.max(ymax,Math.abs(row[off+j]));ymax*=1.08;x.fillStyle='#0e141b';x.fillRect(0,0,w,h);x.font=11*d+'px Segoe UI';x.fillStyle='#9cadbf';x.strokeStyle='#344252';x.textAlign='right';for(let k=-2;k<=2;k++){let y=p.t+(2-k)*(h-p.t-p.b)/4;x.beginPath();x.moveTo(p.l,y);x.lineTo(w-p.r,y);x.stroke();x.fillText((k*ymax/2).toFixed(0),p.l-8*d,y+4*d)}x.textAlign='center';for(let k=0;k<=5;k++){let q=p.l+k*(w-p.l-p.r)/5;x.fillText((tmax*k/5).toFixed(1),q,h-12*d)}x.fillText('simulation time [s]',(p.l+w-p.r)/2,h-2*d);for(let j=0;j<n;j++){x.strokeStyle=colors[j%colors.length];x.lineWidth=1.25*d;x.beginPath();data.rows.forEach((r,i)=>{let q=p.l+r[0]/tmax*(w-p.l-p.r),y=p.t+(ymax-r[off+j])/(2*ymax)*(h-p.t-p.b);i?x.lineTo(q,y):x.moveTo(q,y)});x.stroke()}let q=p.l+Math.min(t,tmax)/tmax*(w-p.l-p.r);x.strokeStyle='#fff';x.lineWidth=1.6*d;x.beginPath();x.moveTo(q,p.t);x.lineTo(q,h-p.b);x.stroke()}function draw(){if(!data.rows.length)return;let r=nearest(video.currentTime),n=data.joints.length;document.getElementById('time').textContent=r[0].toFixed(2)+' s';document.getElementById('height').textContent=r[2].toFixed(3)+' m';document.getElementById('velocity').textContent=r[3].toFixed(2)+' / '+r[4].toFixed(2);for(let i=0;i<n;i++){let t=r[5+i],w=r[5+n+i],p=r[5+2*n+i],l=data.joints[i].effort_limit_nm;document.getElementById('t'+i).textContent=t.toFixed(2)+' N·m';document.getElementById('u'+i).textContent=(100*Math.abs(t)/l).toFixed(1)+'%';document.getElementById('w'+i).textContent=w.toFixed(2)+' rad/s';document.getElementById('p'+i).textContent=p.toFixed(1)+' W'}plot('torque',5,'N·m',video.currentTime);plot('power',5+2*n,'W',video.currentTime);requestAnimationFrame(draw)}selectSkill(location.hash.slice(1) in captures?location.hash.slice(1):'stand');
</script></body></html>'''.replace("__CAPTURES__", encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    captures = {skill: load_capture(root, skill) for skill in ORDER}
    output = root / "index.html"
    output.write_text(build_html(captures), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
