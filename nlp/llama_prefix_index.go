package nlp

// llamaPrefixRadixNode is one compressed token edge in the policy-only prefix
// index. terminal names a complete-cache slot ending at this node; best names
// the most recently used terminal anywhere in the subtree. KV tensors remain in
// the slots and are never stored or shared here.
type llamaPrefixRadixNode struct {
	edge     []int
	children map[int]*llamaPrefixRadixNode
	terminal int
	best     int
}

func (p *LlamaPrefixPool) acquireRadixNode(edge []int) *llamaPrefixRadixNode {
	var node *llamaPrefixRadixNode
	if n := len(p.radixFree); n > 0 {
		node = p.radixFree[n-1]
		p.radixFree = p.radixFree[:n-1]
	} else {
		node = &llamaPrefixRadixNode{children: make(map[int]*llamaPrefixRadixNode)}
	}
	node.edge = edge
	clear(node.children)
	node.terminal, node.best = -1, -1
	return node
}

func (p *LlamaPrefixPool) releaseRadixNode(node *llamaPrefixRadixNode) {
	node.edge = nil
	clear(node.children)
	node.terminal, node.best = -1, -1
	p.radixFree = append(p.radixFree, node)
}

// match reproduces the old linear policy exactly: greatest token LCP, an exact
// terminal ahead of a strict-shorter descendant at equal LCP, then MRU tie-break.
func (p *LlamaPrefixPool) match(salt string, prompt []int) (slot, lcp int, exact bool) {
	node := p.roots[salt]
	if node == nil {
		return -1, 0, false
	}
	pos := 0
	//perfscan:ignore PS3003 radix-tree walk per request not per-token, map inherent
	for pos < len(prompt) {
		child := node.children[prompt[pos]]
		if child == nil {
			return node.best, pos, false
		}
		n := commonTokenPrefix(child.edge, prompt[pos:])
		pos += n
		if n < len(child.edge) {
			return child.best, pos, false
		}
		node = child
	}
	if node.terminal >= 0 {
		return node.terminal, pos, true
	}
	return node.best, pos, false
}

func (p *LlamaPrefixPool) addRadix(slot int) {
	s := &p.slots[slot]
	root := p.roots[s.salt]
	if root == nil {
		root = p.acquireRadixNode(nil)
		p.roots[s.salt] = root
	}
	root.best = slot
	node, pos := root, 0
	//perfscan:ignore PS3003 radix insert per request, tree map inherent
	for pos < len(s.tokens) {
		first := s.tokens[pos]
		child := node.children[first]
		if child == nil {
			leaf := p.acquireRadixNode(s.tokens[pos:])
			leaf.terminal, leaf.best = slot, slot
			node.children[first] = leaf
			return
		}
		n := commonTokenPrefix(child.edge, s.tokens[pos:])
		if n == len(child.edge) {
			child.best = slot
			node, pos = child, pos+n
			continue
		}

		mid := p.acquireRadixNode(append([]int(nil), child.edge[:n]...))
		oldSuffix := child.edge[n:]
		child.edge = oldSuffix
		mid.children[oldSuffix[0]] = child
		mid.best = slot
		node.children[first] = mid
		pos += n
		if pos == len(s.tokens) {
			mid.terminal = slot
			return
		}
		leaf := p.acquireRadixNode(s.tokens[pos:])
		leaf.terminal, leaf.best = slot, slot
		mid.children[leaf.edge[0]] = leaf
		return
	}
	if node.terminal >= 0 {
		panic("nlp: duplicate LlamaPrefixPool radix terminal")
	}
	node.terminal, node.best = slot, slot
}

func (p *LlamaPrefixPool) touchRadix(slot int) {
	s := &p.slots[slot]
	node := p.roots[s.salt]
	if node == nil {
		panic("nlp: missing LlamaPrefixPool radix root")
	}
	node.best = slot
	pos := 0
	//perfscan:ignore PS3003 radix touch per request, map inherent
	for pos < len(s.tokens) {
		child := node.children[s.tokens[pos]]
		if child == nil || commonTokenPrefix(child.edge, s.tokens[pos:]) != len(child.edge) {
			panic("nlp: missing LlamaPrefixPool radix path")
		}
		pos += len(child.edge)
		child.best = slot
		node = child
	}
	if node.terminal != slot {
		panic("nlp: mismatched LlamaPrefixPool radix terminal")
	}
}

func (p *LlamaPrefixPool) removeRadix(slot int) {
	s := &p.slots[slot]
	root := p.roots[s.salt]
	if root == nil || !p.removeRadixFrom(root, s.tokens, 0, slot) {
		panic("nlp: missing LlamaPrefixPool radix removal path")
	}
	if root.terminal < 0 && len(root.children) == 0 {
		delete(p.roots, s.salt)
		p.releaseRadixNode(root)
	}
}

// removeRadixFrom returns whether slot was found and removed. Nodes are merged
// on unwind so every non-root, non-terminal unary path remains one radix edge.
func (p *LlamaPrefixPool) removeRadixFrom(node *llamaPrefixRadixNode, tokens []int, pos, slot int) bool {
	if pos == len(tokens) {
		if node.terminal != slot {
			return false
		}
		node.terminal = -1
	} else {
		child := node.children[tokens[pos]]
		if child == nil || commonTokenPrefix(child.edge, tokens[pos:]) != len(child.edge) {
			return false
		}
		if !p.removeRadixFrom(child, tokens, pos+len(child.edge), slot) {
			return false
		}
		if child.terminal < 0 && len(child.children) == 0 {
			delete(node.children, tokens[pos])
			p.releaseRadixNode(child)
		} else if child.terminal < 0 && len(child.children) == 1 {
			var grand *llamaPrefixRadixNode
			for _, grand = range child.children {
			}
			merged := make([]int, 0, len(child.edge)+len(grand.edge))
			merged = append(merged, child.edge...)
			merged = append(merged, grand.edge...)
			child.edge = merged
			child.children, grand.children = grand.children, child.children
			child.terminal = grand.terminal
			child.best = grand.best
			p.releaseRadixNode(grand)
		}
	}
	p.recomputeRadixBest(node)
	return true
}

func (p *LlamaPrefixPool) recomputeRadixBest(node *llamaPrefixRadixNode) {
	best := node.terminal
	for _, child := range node.children {
		if p.newerSlot(child.best, best) {
			best = child.best
		}
	}
	node.best = best
}

func (p *LlamaPrefixPool) newerSlot(a, b int) bool {
	if a < 0 {
		return false
	}
	if b < 0 {
		return true
	}
	if p.slots[a].used != p.slots[b].used {
		return p.slots[a].used > p.slots[b].used
	}
	return a < b
}

func (p *LlamaPrefixPool) linkMRU(slot int) {
	p.slots[slot].prev = p.mru
	p.slots[slot].next = -1
	if p.mru >= 0 {
		p.slots[p.mru].next = slot
	} else {
		p.lru = slot
	}
	p.mru = slot
}

func (p *LlamaPrefixPool) unlinkLRU(slot int) {
	prev, next := p.slots[slot].prev, p.slots[slot].next
	if prev >= 0 {
		p.slots[prev].next = next
	} else {
		p.lru = next
	}
	if next >= 0 {
		p.slots[next].prev = prev
	} else {
		p.mru = prev
	}
	p.slots[slot].prev, p.slots[slot].next = -1, -1
}

func (p *LlamaPrefixPool) moveToMRU(slot int) {
	if slot == p.mru {
		return
	}
	p.unlinkLRU(slot)
	p.linkMRU(slot)
}
