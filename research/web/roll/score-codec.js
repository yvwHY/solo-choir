(() => {
  const VOICES = ["Melody","Sop","Alto","Tenor","Bass"];
  const QMS = 20, Q = QMS/1000;
  const seqOf = (s,v) => v==="Melody" ? (s.melody||[]) : ((s.voices&&s.voices[v])||[]);
  const noteCount = s => VOICES.reduce((n,v)=>n+seqOf(s,v).length,0);
  const durationSec = s => Math.max(0,...VOICES.flatMap(v=>seqOf(s,v).map(n=>n.start+n.dur)));

  // --- varint (unsigned LEB128) ---
  function putV(arr,n){ n=Math.max(0,Math.round(n)); do{ let b=n&0x7f; n>>>=7; arr.push(n?b|0x80:b);}while(n); }
  function getV(buf,st){ let n=0,sh=0,b; do{ b=buf[st.i++]; n|=(b&0x7f)<<sh; sh+=7; }while(b&0x80); return n>>>0; }

  function toBytes(s){
    const out=[1, ((s.key&&s.key.root)|0) & 0xff, QMS];
    for(const v of VOICES){
      const ns=[...seqOf(s,v)].sort((a,b)=>a.start-b.start);
      putV(out, ns.length); let prev=0;
      for(const n of ns){ const t=Math.round(n.start/Q); putV(out,t-prev); putV(out,Math.round(n.dur/Q)); out.push(((n.midi|0)%128+128)%128); prev=t; }
    }
    return Uint8Array.from(out);
  }
  function fromBytes(buf){
    const st={i:0}; const ver=buf[st.i++]; const root=buf[st.i++]; const q=(buf[st.i++]||20)/1000;
    const s={key:{root}, melody:[], voices:{Sop:[],Alto:[],Tenor:[],Bass:[]}};
    for(const v of VOICES){ const c=getV(buf,st); let prev=0; const dst = v==="Melody"?s.melody:s.voices[v];
      for(let k=0;k<c;k++){ prev+=getV(buf,st); const dur=getV(buf,st)*q; const midi=buf[st.i++]; dst.push({start:prev*q,dur,midi}); } }
    return s;
  }

  // --- deflate-raw via CompressionStream (pywebview macOS WebKit + mobile Safari/Chrome) ---
  async function deflate(u8){ const cs=new CompressionStream("deflate-raw"); const w=cs.writable.getWriter(); w.write(u8); w.close();
    return new Uint8Array(await new Response(cs.readable).arrayBuffer()); }
  async function inflate(u8){ const ds=new DecompressionStream("deflate-raw"); const w=ds.writable.getWriter(); w.write(u8); w.close();
    return new Uint8Array(await new Response(ds.readable).arrayBuffer()); }

  // --- base64url (URL-safe AND QR-safe; no %, space, or escaping needed) ---
  function b64u(u8){ let s=""; for(let i=0;i<u8.length;i++) s+=String.fromCharCode(u8[i]);
    return btoa(s).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,""); }
  function b64ud(str){ str=str.replace(/-/g,"+").replace(/_/g,"/"); while(str.length%4) str+="=";
    const s=atob(str); const u=new Uint8Array(s.length); for(let i=0;i<s.length;i++) u[i]=s.charCodeAt(i); return u; }

  async function encodeScore(s){ return b64u(await deflate(toBytes(s))); }
  async function decodeScore(p){ return fromBytes(await inflate(b64ud(p))); }

  const api={ encodeScore, decodeScore, VOICES, noteCount, durationSec };
  if(typeof window!=="undefined") window.ScoreCodec=api;
  if(typeof module!=="undefined" && module.exports) module.exports=api;
})();
