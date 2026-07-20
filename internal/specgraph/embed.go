package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// Embedder turns text into one sentence vector. The production implementation
// is goai's own inference stack (bertEmbedder — dogfooding, no external API,
// no cgo); tests inject a deterministic fake.
type Embedder interface {
	Embed(text string) ([]float64, error)
}

// bertEmbedder is BertFromHF → Forward → MeanPool over a local HF checkpoint
// directory (config.json + model.safetensors + tokenizer.json), e.g. a
// sentence-transformers MiniLM/bge-small download. Model path comes from
// -embed-model or SPECGRAPH_EMBED_MODEL; without one, search stays BM25-only.
type bertEmbedder struct {
	model *nlp.Bert
	tok   *nlp.WordPiece
}

// maxEmbedTokens caps the encoder input (BERT position table is 512).
const maxEmbedTokens = 512

func newBertEmbedder(dir string) (*bertEmbedder, error) {
	cfgJSON, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("specgraph: embed model config: %w", err)
	}
	cfg, err := nlp.BertConfigFromHF(cfgJSON)
	if err != nil {
		return nil, err
	}
	ts, _, err := safetensors.LoadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("specgraph: embed model weights: %w", err)
	}
	model, err := nlp.BertFromHF(ts, cfg)
	if err != nil {
		return nil, err
	}
	tokJSON, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("specgraph: embed model tokenizer: %w", err)
	}
	tok, err := nlp.WordPieceFromJSON(tokJSON)
	if err != nil {
		return nil, err
	}
	return &bertEmbedder{model: model, tok: tok}, nil
}

func (e *bertEmbedder) Embed(text string) ([]float64, error) {
	ids := e.tok.Encode(text)
	if len(ids) == 0 {
		return nil, fmt.Errorf("specgraph: empty tokenization")
	}
	if len(ids) > maxEmbedTokens {
		ids = ids[:maxEmbedTokens]
	}
	hidden, err := e.model.Forward(backend.NewContext(), ids, nil)
	if err != nil {
		return nil, err
	}
	return nlp.MeanPool(hidden, nil)
}

// embedRecord is one line of .specgraph/embeddings.jsonl: node id, the
// content hash the vector was computed from, and the vector. Keyed on the
// hash so `index` only re-embeds changed nodes.
type embedRecord struct {
	ID   string    `json:"id"`
	Hash string    `json:"hash"`
	Vec  []float64 `json:"vec"`
}

// nodeHash is the embed cache key: kind + text (+fix), so a reworded row
// re-embeds and an untouched one never does.
func nodeHash(n *Node) string {
	h := sha256.Sum256([]byte(string(n.Kind) + "\x00" + n.Text + "\x00" + n.Meta["fix"]))
	return hex.EncodeToString(h[:8])
}

// loadEmbeddings reads the vector cache (missing file = empty map).
func loadEmbeddings(path string) map[string]embedRecord {
	out := map[string]embedRecord{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		var r embedRecord
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.ID != "" {
			out[r.ID] = r
		}
	}
	return out
}

// saveEmbeddings writes the vector cache atomically, sorted by id.
func saveEmbeddings(path string, recs map[string]embedRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	ids := make([]string, 0, len(recs))
	for id := range recs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	for _, id := range ids {
		b, err := json.Marshal(recs[id])
		if err != nil {
			f.Close()
			return err
		}
		w.Write(b)
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// indexEmbeddings (re-)embeds every embeddable node whose hash changed and
// prunes vectors for vanished nodes. Returns (embedded, kept, total).
func indexEmbeddings(g *Graph, emb Embedder, path string) (int, int, int, error) {
	recs := loadEmbeddings(path)
	fresh := map[string]embedRecord{}
	embedded, kept := 0, 0
	for _, id := range g.NodeIDs() {
		n := g.Nodes[id]
		if n.Text == "" {
			continue
		}
		h := nodeHash(n)
		if r, ok := recs[id]; ok && r.Hash == h {
			fresh[id] = r
			kept++
			continue
		}
		vec, err := emb.Embed(embedText(n))
		if err != nil {
			return embedded, kept, len(fresh), fmt.Errorf("specgraph: embedding %s: %w", id, err)
		}
		fresh[id] = embedRecord{ID: id, Hash: h, Vec: vec}
		embedded++
	}
	return embedded, kept, len(fresh), saveEmbeddings(path, fresh)
}

// embedText is what gets embedded per node: id, kind, and the text (plus a
// bug's fix — the cure describes the pattern as much as the cause).
func embedText(n *Node) string {
	s := n.ID + " " + string(n.Kind) + ": " + n.Text
	if fix := n.Meta["fix"]; fix != "" {
		s += " fix: " + fix
	}
	return truncate(s, 2000)
}

// vectorSearch scores the query vector against every cached node vector by
// cosine (nlp.CosineRerank — brute force is <10ms at this corpus size).
func vectorSearch(recs map[string]embedRecord, query []float64, k int) ([]scored, error) {
	ids := make([]string, 0, len(recs))
	for id := range recs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	cands := make([][]float64, len(ids))
	for i, id := range ids {
		cands[i] = recs[id].Vec
	}
	ranked, err := nlp.CosineRerank(query, cands)
	if err != nil {
		return nil, err
	}
	if k > 0 && len(ranked) > k {
		ranked = ranked[:k]
	}
	out := make([]scored, len(ranked))
	for i, r := range ranked {
		out[i] = scored{ID: ids[r.Index], Score: r.Score}
	}
	return out, nil
}
