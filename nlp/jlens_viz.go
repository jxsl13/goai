package nlp

import (
	"encoding/json"
	"fmt"
	"html"
	"sort"
	"strconv"
	"strings"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// J-lens slice visualisation (§T814, §R250): one SELF-CONTAINED HTML page
// rendering the layer × position lens grid of [JLens.Slice] — rows are
// residual layers top→bottom with the final row being the model's actual
// next-token output, columns are token positions, and every cell shows the
// lens top-1 token with the vocab rank of the model's real output as a
// superscript. Clicking a cell pins its token: pinned tokens get distinct
// colors, a rank-tracking line chart (rank across layers at the pinned
// position, inline SVG), and a rank heatmap overlay tinting every cell by the
// active pinned token's rank there. The §T813 J-space occupancy can be added
// as a per-cell corner badge ([WithJLensHTMLOccupancy]).
//
// Unlike the reference implementation's d3-based page, the emitted page is
// ZERO-DEPENDENCY by construction (§C2): all CSS, JS, and SVG are hand-rolled
// inline, there are no src/href attributes, no CDN, no fonts, no fetches —
// nothing that would touch the network. All token text is HTML-escaped
// server-side (grid) and escaped again client-side before any innerHTML use.

// jlensHTMLOpts collects the JLensHTML knobs (functional options).
type jlensHTMLOpts struct {
	title string
	text  func(id int) string
	occK  int
}

// JLensHTMLOption configures [JLensHTML].
type JLensHTMLOption func(*jlensHTMLOpts)

// WithJLensHTMLTitle sets the page title (default "J-lens slice").
func WithJLensHTMLTitle(title string) JLensHTMLOption {
	return func(o *jlensHTMLOpts) { o.title = title }
}

// WithJLensHTMLTokenText supplies the token-id → display-text mapping
// (tokenizer-agnostic: pass your tokenizer's single-token decode). The
// default renders "t<id>". Text is escaped before it reaches the page, so
// tokens containing '<', '>', '&' or quotes are safe.
func WithJLensHTMLTokenText(fn func(id int) string) JLensHTMLOption {
	return func(o *jlensHTMLOpts) { o.text = fn }
}

// WithJLensHTMLOccupancy enables the J-space occupancy overlay (§T813): every
// cell gets a corner badge with the occupancy of that cell's residual under
// [JSpaceDecompose] with at most k directions (k ≤ 0 selects the default 25).
func WithJLensHTMLOccupancy(k int) JLensHTMLOption {
	return func(o *jlensHTMLOpts) {
		if k <= 0 {
			k = 25
		}
		o.occK = k
	}
}

// jlensPageData is the machine-readable payload embedded in the page as
// application/json — the [JLensSlice] grid verbatim plus the rank tensors the
// interactive charts need. Field-for-field: Layers[i]/Top[i]/Rank[i] mirror
// Slice().Rows[i], Output mirrors Slice().Output.
type jlensPageData struct {
	Title   string            `json:"title"`
	Tokens  []int             `json:"tokens"`
	Text    map[string]string `json:"text"`
	Layers  []int             `json:"layers"`
	Top     [][]int           `json:"top"`
	Rank    [][]int           `json:"rank"`
	Output  []int             `json:"output"`
	Tracked []int             `json:"tracked"`
	// Ranks maps each tracked token id to its [rows][positions] vocab rank
	// under every cell's readout — the data behind pin charts and the rank
	// heatmap (the reference's ranks/{tid}.bin sidecars, embedded).
	Ranks map[string][][]int `json:"ranks"`
	Vocab int                `json:"vocab"`
	Occ   [][]int            `json:"occ,omitempty"`
}

// JLensHTML renders the full lens view of a prompt as one self-contained HTML
// page (§T814): it computes [JLens.Slice] on model (one forward pass with
// residual capture per pass), the per-cell vocab ranks of every pinnable
// token, optionally the §T813 occupancy grid, and emits the page described in
// the package comment above. The returned string is a complete document —
// write it to a file and open it in any browser; it performs zero network
// requests.
//
// The embedded JSON payload (script#jlv-data) carries the [JLensSlice] grid
// verbatim (layers/top/rank/output) so downstream tooling can consume the
// page as data, and the tracked-token rank tensors that power the
// interactivity. Determinism: for a fixed model, lens, tokens and options the
// output is byte-identical across runs (all embedded numbers are integers;
// map keys are emitted in sorted order).
func JLensHTML(model LensReadoutModel, lens *JLens, tokens []int, opts ...JLensHTMLOption) (string, error) {
	o := jlensHTMLOpts{
		title: "J-lens slice",
		text:  func(id int) string { return fmt.Sprintf("t%d", id) },
	}
	for _, opt := range opts {
		opt(&o)
	}
	if len(tokens) == 0 {
		return "", fmt.Errorf("nlp: JLensHTML: empty token sequence")
	}
	ctx := backend.NewContext()
	sl, err := lens.Slice(model, ctx, tokens)
	if err != nil {
		return "", fmt.Errorf("nlp: JLensHTML: %w", err)
	}
	resids := make([]*tensor.Tensor, 0, lens.Layers+1)
	if _, err := model.ForwardResiduals(ctx, tokens, func(l int, h *tensor.Tensor) {
		resids = append(resids, h)
	}); err != nil {
		return "", fmt.Errorf("nlp: JLensHTML: %w", err)
	}
	T, R := len(tokens), len(sl.Rows)

	// Pinnable tokens: everything that appears in the grid or output row.
	seen := map[int]bool{}
	for _, row := range sl.Rows {
		for _, tid := range row.Top {
			seen[tid] = true
		}
	}
	for _, tid := range sl.Output {
		seen[tid] = true
	}
	tracked := make([]int, 0, len(seen))
	for tid := range seen {
		tracked = append(tracked, tid)
	}
	sort.Ints(tracked)

	data := jlensPageData{
		Title:   o.title,
		Tokens:  tokens,
		Text:    map[string]string{},
		Output:  sl.Output,
		Tracked: tracked,
		Ranks:   make(map[string][][]int, len(tracked)),
	}
	for _, tid := range tracked {
		grid := make([][]int, R)
		for r := range grid {
			grid[r] = make([]int, T)
		}
		data.Ranks[strconv.Itoa(tid)] = grid
	}
	if o.occK > 0 {
		data.Occ = make([][]int, R)
	}
	for ri, row := range sl.Rows {
		data.Layers = append(data.Layers, row.Layer)
		data.Top = append(data.Top, row.Top)
		data.Rank = append(data.Rank, row.Rank)
		logits, err := lens.Apply(model, ctx, resids[row.Layer], row.Layer)
		if err != nil {
			return "", fmt.Errorf("nlp: JLensHTML: layer %d readout: %w", row.Layer, err)
		}
		data.Vocab = logits.Shape()[1]
		for p := 0; p < T; p++ {
			ro := newJLensReadout(logits, p)
			for _, tid := range tracked {
				data.Ranks[strconv.Itoa(tid)][ri][p] = ro.RankOf(tid)
			}
		}
		if o.occK > 0 {
			data.Occ[ri] = make([]int, T)
			for p := 0; p < T; p++ {
				h := tensor.New(tensor.F64, tensor.Shape{lens.Dim})
				for b := 0; b < lens.Dim; b++ {
					h.SetF64(resids[row.Layer].AtF64(p, b), b)
				}
				dec, err := JSpaceDecompose(lens, model, h, row.Layer, o.occK)
				if err != nil {
					return "", fmt.Errorf("nlp: JLensHTML: occupancy at layer %d pos %d: %w", row.Layer, p, err)
				}
				data.Occ[ri][p] = dec.Occupancy
			}
		}
	}
	for _, tid := range tokens {
		data.Text[strconv.Itoa(tid)] = o.text(tid)
	}
	for _, tid := range tracked {
		data.Text[strconv.Itoa(tid)] = o.text(tid)
	}
	blob, err := json.Marshal(&data) // escapes <, >, & → <…: safe inside <script>
	if err != nil {
		return "", fmt.Errorf("nlp: JLensHTML: marshal payload: %w", err)
	}

	var b strings.Builder
	esc := html.EscapeString
	b.WriteString("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n<title>")
	b.WriteString(esc(o.title))
	b.WriteString("</title>\n<style>\n" + jlensVizCSS + "</style></head><body>\n<div class=\"jlv\">\n")
	fmt.Fprintf(&b, "<h1>%s</h1>\n", esc(o.title))
	fmt.Fprintf(&b, "<div class=\"jlv-meta\">%d layer rows × %d positions · vocab %d · click a cell to pin its token; a pin gets a rank chart and tints the grid by its rank</div>\n", R, T, data.Vocab)
	fmt.Fprintf(&b, "<div class=\"jlv-grid\" id=\"jlvGrid\" data-rows=\"%d\" data-cols=\"%d\" style=\"grid-template-columns:max-content repeat(%d,minmax(56px,max-content))\">\n", R, T, T)
	b.WriteString("<div class=\"jlv-corner\"></div>")
	for p, tid := range tokens {
		fmt.Fprintf(&b, "<div class=\"jlv-colhead\"><span class=\"jlv-tok\">%s</span><span class=\"jlv-pos\">%d</span></div>", esc(o.text(tid)), p)
	}
	b.WriteString("\n")
	for ri, row := range sl.Rows {
		label := fmt.Sprintf("L%d", row.Layer)
		if row.Layer == lens.Layers {
			label = "out" // identity transport: the model's actual output row
		}
		fmt.Fprintf(&b, "<div class=\"jlv-rowhead\">%s</div>", label)
		for p := 0; p < T; p++ {
			cls := "jlv-cell"
			if row.Rank[p] == 0 {
				cls += " jlv-agree"
			}
			fmt.Fprintf(&b, "<div class=\"%s\" data-r=\"%d\" data-p=\"%d\" data-tid=\"%d\"><span class=\"jlv-tok\">%s</span><sup class=\"jlv-rank\">%d</sup>",
				cls, ri, p, row.Top[p], esc(o.text(row.Top[p])), row.Rank[p])
			if data.Occ != nil {
				fmt.Fprintf(&b, "<span class=\"jlv-occ\" title=\"J-space occupancy\">%d</span>", data.Occ[ri][p])
			}
			b.WriteString("</div>")
		}
		b.WriteString("\n")
	}
	b.WriteString("</div>\n<div class=\"jlv-pins\" id=\"jlvPins\"></div>\n<div class=\"jlv-chart\" id=\"jlvChart\"></div>\n</div>\n")
	b.WriteString("<script id=\"jlv-data\" type=\"application/json\">")
	b.Write(blob)
	b.WriteString("</script>\n<script>\n" + jlensVizJS + "</script>\n</body></html>\n")
	return b.String(), nil
}

// jlensVizCSS is the page stylesheet: pure inline CSS, no url()/@import — the
// zero-external-request discipline is asserted by the §T814 tests.
const jlensVizCSS = `body{margin:12px;font:13px/1.4 system-ui,sans-serif;color:#333;background:#fff}
.jlv h1{font-size:16px;margin:0 0 2px}
.jlv-meta{color:#777;font-size:11px;margin:0 0 8px}
.jlv-grid{display:grid;gap:1px;background:#ddd;border:1px solid #ddd;width:max-content;max-width:100%;overflow-x:auto}
.jlv-grid>div{background:#fff;padding:2px 6px}
.jlv-corner{background:#fafafa}
.jlv-colhead{background:#fafafa;text-align:center;font:11px monospace}
.jlv-colhead .jlv-pos{display:block;color:#999;font-size:9px}
.jlv-rowhead{background:#fafafa;font:11px monospace;color:#666;text-align:right}
.jlv-cell{position:relative;font:12px monospace;cursor:pointer;padding-right:14px}
.jlv-cell:hover{outline:1px solid #999}
.jlv-rank{color:#b00;font-size:9px;margin-left:1px}
.jlv-agree .jlv-rank{color:#bbb}
.jlv-occ{position:absolute;top:0;right:0;font-size:8px;color:#fff;background:#7a6ea8;border-radius:0 0 0 3px;padding:0 3px;line-height:10px}
.jlv-pins{margin:8px 0;min-height:22px}
.jlv-pins .jlv-dim{color:#999;font-size:11px}
.jlv-chip{font:11px monospace;border:2px solid;border-radius:3px;background:#fff;padding:1px 6px;margin-right:6px;cursor:pointer}
.jlv-chart svg{background:#fff;border:1px solid #eee}
`

// jlensVizJS is the page's interactivity: hand-rolled vanilla JS (no
// libraries, no network). SVG is created via innerHTML so no namespace URL
// string ever appears in the page. Comments deliberately use block form only,
// keeping the emitted page free of "//" sequences.
const jlensVizJS = `(function(){
"use strict";
var D=JSON.parse(document.getElementById("jlv-data").textContent);
var R=D.layers.length,T=D.tokens.length,V=Math.max(2,D.vocab);
var COLORS=["#e15759","#4e79a7","#59a14f","#edc948","#b07aa1","#76b7b2","#ff9da7","#9c755f"];
var pins=[]; /* {tid,pos,color} in pin order; last pin drives the heatmap */
var cells=Array.prototype.slice.call(document.querySelectorAll(".jlv-cell"));
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function tokStr(t){var s=D.text[String(t)];return s===undefined?("["+t+"]"):s;}
function rankOf(t,r,p){var a=D.ranks[String(t)];return a?a[r][p]:-1;}
function pinColor(t){for(var i=0;i<pins.length;i++){if(pins[i].tid===t){return pins[i].color;}}return null;}
function togglePin(t,p){
  for(var i=0;i<pins.length;i++){if(pins[i].tid===t){pins.splice(i,1);render();return;}}
  var used={},c=null;
  pins.forEach(function(q){used[q.color]=1;});
  for(var j=0;j<COLORS.length;j++){if(!used[COLORS[j]]){c=COLORS[j];break;}}
  pins.push({tid:t,pos:p,color:c||COLORS[pins.length%COLORS.length]});
  render();
}
cells.forEach(function(el){
  el.addEventListener("click",function(){togglePin(+el.dataset.tid,+el.dataset.p);});
});
function renderPins(){
  var el=document.getElementById("jlvPins");
  if(!pins.length){el.innerHTML="<span class=\"jlv-dim\">no pins - click a grid cell to pin its top-1 token</span>";return;}
  el.innerHTML=pins.map(function(q){
    return "<button class=\"jlv-chip\" data-tid=\""+q.tid+"\" style=\"border-color:"+q.color+"\">"+
      esc(tokStr(q.tid))+" @"+q.pos+" &times;</button>";
  }).join("");
}
document.getElementById("jlvPins").addEventListener("click",function(e){
  var b=e.target;
  while(b&&b!==this&&!b.dataset.tid){b=b.parentNode;}
  if(b&&b.dataset&&b.dataset.tid){togglePin(+b.dataset.tid,0);}
});
function renderHeat(){
  var act=pins.length?pins[pins.length-1]:null;
  cells.forEach(function(el){
    var pc=pinColor(+el.dataset.tid);
    el.style.outline=pc?("2px solid "+pc):"";
    if(!act){el.style.background="";return;}
    var rk=rankOf(act.tid,+el.dataset.r,+el.dataset.p);
    if(rk<0){el.style.background="";return;}
    var t=1-Math.log(rk+1)/Math.log(V);
    var a=Math.round(16+180*Math.max(0,t)).toString(16);
    if(a.length<2){a="0"+a;}
    el.style.background=act.color+a;
  });
}
function renderChart(){
  var el=document.getElementById("jlvChart");
  if(!pins.length){el.innerHTML="";return;}
  var W=Math.max(340,90+R*70),H=190,pl=40,pt=12,pb=28,pr=110;
  var iw=W-pl-pr,ih=H-pt-pb,lg=Math.log(V);
  function xf(i){return pl+(R<2?iw/2:i*iw/(R-1));}
  function yf(rk){return pt+Math.log(rk+1)/lg*ih;}
  var s="<svg width=\""+W+"\" height=\""+H+"\">";
  s+="<rect x=\""+pl+"\" y=\""+pt+"\" width=\""+iw+"\" height=\""+ih+"\" fill=\"#f5f5f5\"/>";
  for(var i=0;i<R;i++){
    var lab=(D.layers[i]===undefined)?i:D.layers[i];
    if(i===R-1){lab="out";}else{lab="L"+lab;}
    s+="<line x1=\""+xf(i)+"\" y1=\""+pt+"\" x2=\""+xf(i)+"\" y2=\""+(pt+ih)+"\" stroke=\"#fff\"/>";
    s+="<text x=\""+xf(i)+"\" y=\""+(H-10)+"\" text-anchor=\"middle\" font-size=\"9\" fill=\"#666\">"+lab+"</text>";
  }
  s+="<text x=\""+(pl-4)+"\" y=\""+(pt+8)+"\" text-anchor=\"end\" font-size=\"9\" fill=\"#666\">1</text>";
  s+="<text x=\""+(pl-4)+"\" y=\""+(pt+ih)+"\" text-anchor=\"end\" font-size=\"9\" fill=\"#666\">"+V+"</text>";
  s+="<text x=\""+(pl-28)+"\" y=\""+(pt+ih/2)+"\" font-size=\"9\" fill=\"#999\" transform=\"rotate(-90 "+(pl-28)+" "+(pt+ih/2)+")\" text-anchor=\"middle\">rank</text>";
  pins.forEach(function(q,qi){
    var pts=[];
    for(var r=0;r<R;r++){
      var rk=rankOf(q.tid,r,q.pos);
      if(rk>=0){pts.push(xf(r)+","+yf(rk));}
    }
    if(pts.length){
      s+="<polyline points=\""+pts.join(" ")+"\" fill=\"none\" stroke=\""+q.color+"\" stroke-width=\"2\"/>";
    }
    s+="<text x=\""+(W-pr+8)+"\" y=\""+(pt+12+qi*13)+"\" font-size=\"10\" fill=\""+q.color+"\" font-family=\"monospace\">"+esc(tokStr(q.tid))+" @"+q.pos+"</text>";
  });
  s+="</svg>";
  el.innerHTML=s;
}
function render(){renderPins();renderHeat();renderChart();}
render();
})();
`
