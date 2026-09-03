"""One-off verification for the head/layout work (UI_BUILD_SPEC §6 step 1).

Loads the UI through shell.py's static server, then:
  - grabs the actual WebGL canvas via toDataURL (bypasses the screen-recording
    permission) for panel-open and panel-collapsed states -> app/_shot_*.png
  - probes head on-screen boxes + the panel rect so we can confirm Bass is
    clickable when the panel is collapsed.
Not part of the app.

    /opt/anaconda3/envs/vcclient-dev/bin/python app/_smoke_test.py
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import webview

from shell import REPO_ROOT, UI_REL, start_static_server

OPEN_PNG = REPO_ROOT / "app" / "_shot_open.png"
CLOSED_PNG = REPO_ROOT / "app" / "_shot_closed.png"

CAPTURE = r"""
(function(){try{
  renderer.setClearColor(0xb4c0cb,1); renderer.render(scene,camera);
  var u=renderer.domElement.toDataURL('image/png');
  renderer.setClearColor(0x000000,0); renderer.render(scene,camera);
  return u;
}catch(e){return 'ERR:'+e;}})()
"""

PROBE = r"""
(function(){
  try{
    function box(grp){
      var s=grp.children[0]; s.updateWorldMatrix(true,true);
      var g=s.geometry; g.computeBoundingBox(); var bb=g.boundingBox;
      var mnx=1e9,mny=1e9,mxx=-1e9,mxy=-1e9;
      for(var xi=0;xi<2;xi++)for(var yi=0;yi<2;yi++)for(var zi=0;zi<2;zi++){
        var v=new THREE.Vector3(xi?bb.max.x:bb.min.x,yi?bb.max.y:bb.min.y,zi?bb.max.z:bb.min.z);
        v.applyMatrix4(s.matrixWorld); v.project(camera);
        var px=(v.x*0.5+0.5)*innerWidth, py=(-v.y*0.5+0.5)*innerHeight;
        mnx=Math.min(mnx,px);mxx=Math.max(mxx,px);mny=Math.min(mny,py);mxy=Math.max(mxy,py);
      }
      return {minx:Math.round(mnx),maxx:Math.round(mxx),miny:Math.round(mny),maxy:Math.round(mxy)};
    }
    var p=document.getElementById('voicePanel');
    var r=p.getBoundingClientRect();
    var collapsed=p.classList.contains('collapsed');
    var pr=Math.round(r.right);
    var heads=pickables.map(function(P){var b=box(P.grp);
      return {name:P.part.name,minx:b.minx,maxx:b.maxx,hFrac:+((b.maxy-b.miny)/innerHeight).toFixed(2),
              clearsPanel:(collapsed||pr<=0)?true:(b.minx>pr),
              onScreen:(b.minx>=0&&b.maxx<=innerWidth&&b.miny>=0&&b.maxy<=innerHeight)};});
    return JSON.stringify({badge:(document.getElementById('modelBadge')||{}).textContent,
      win:innerWidth+'x'+innerHeight,collapsed:collapsed,panelRight:pr,heads:heads});
  }catch(e){return JSON.stringify({error:String(e)});}
})()
"""


def save_png(data_url: str, path: Path) -> str:
    if not data_url or not data_url.startswith("data:image/png;base64,"):
        return f"capture failed: {str(data_url)[:60]}"
    path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
    return f"saved {path.name} ({path.stat().st_size//1024} KB)"


def print_probe(window: "webview.Window", tag: str) -> None:
    info = json.loads(window.evaluate_js(PROBE))
    print(f"--- probe [{tag}] badge={info.get('badge')!r} win={info.get('win')} "
          f"collapsed={info.get('collapsed')} panelRight={info.get('panelRight')}")
    for h in info.get("heads", []):
        ok = h["clearsPanel"] and h["onScreen"] and h["hFrac"] >= 0.30
        print(f"    [{'OK' if ok else '!!'}] {h['name']:5s} x[{h['minx']:>5},{h['maxx']:>5}] "
              f"hFrac={h['hFrac']} clears={h['clearsPanel']} onScreen={h['onScreen']}")


def checker(window: "webview.Window") -> None:
    time.sleep(4.0)
    print(save_png(window.evaluate_js(CAPTURE), OPEN_PNG))
    print_probe(window, "panel open")

    window.evaluate_js("document.getElementById('panelClose').click()")
    time.sleep(0.7)
    print(save_png(window.evaluate_js(CAPTURE), CLOSED_PNG))
    print_probe(window, "panel collapsed")
    window.destroy()


def main() -> None:
    base = start_static_server(REPO_ROOT)
    win = webview.create_window("Solo Choir (smoke test)", url=f"{base}/{UI_REL}",
                                width=1280, height=820, background_color="#c6d0d9")
    webview.start(checker, win)


if __name__ == "__main__":
    main()
