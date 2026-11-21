class_name Foam
extends RefCounted
## A foam: cells (one site each), edges between neighbouring cells, every edge in exactly one
## group, two edges of a group never sharing a cell. A move = (group, colours a and b): every
## pair of the group holding a and b swaps them. Same rule as model/mosaic_model/foam.py.

var aspect := 1.0
var sites := PackedVector2Array()  ## in [0, aspect] x [0, 1], y down
var edge_u := PackedInt32Array()
var edge_v := PackedInt32Array()
var edge_group := PackedInt32Array()
var n_cells := 0
var n_groups := 0
var pairs_u: Array[PackedInt32Array] = []  ## per group, the pairs (u, v)
var pairs_v: Array[PackedInt32Array] = []
var neighbours: Array[PackedInt32Array] = []  ## per cell
var _group_of := {}  ## u * n_cells + v (u < v) -> group


## Builds the foam from level data; returns "" or what is wrong.
func setup(p_aspect: float, p_sites: Array, p_edges: Array, p_group: Array) -> String:
	aspect = p_aspect
	n_cells = p_sites.size()
	sites.resize(n_cells)
	for i in n_cells:
		sites[i] = Vector2(p_sites[i][0], p_sites[i][1])
	if p_edges.size() != p_group.size():
		return "edges and groups differ in number"
	n_groups = 0
	for g in p_group:
		n_groups = maxi(n_groups, int(g) + 1)
	pairs_u.clear()
	pairs_v.clear()
	neighbours.clear()
	_group_of.clear()
	for g in n_groups:
		pairs_u.append(PackedInt32Array())
		pairs_v.append(PackedInt32Array())
	for i in n_cells:
		neighbours.append(PackedInt32Array())
	var seen := {}  # g * n_cells + cell: a cell already used by group g
	for e in p_edges.size():
		var u := int(p_edges[e][0])
		var v := int(p_edges[e][1])
		var g := int(p_group[e])
		if u == v or u < 0 or v < 0 or u >= n_cells or v >= n_cells:
			return "edge %d outside the cells" % e
		if u > v:
			var t := u
			u = v
			v = t
		if seen.has(g * n_cells + u) or seen.has(g * n_cells + v):
			return "two pairs of group %d share a cell" % g
		seen[g * n_cells + u] = true
		seen[g * n_cells + v] = true
		edge_u.append(u)
		edge_v.append(v)
		edge_group.append(g)
		pairs_u[g].append(u)
		pairs_v[g].append(v)
		neighbours[u].append(v)
		neighbours[v].append(u)
		_group_of[u * n_cells + v] = g
	return ""


## Group of the edge between two cells, -1 if they are not neighbours.
func group_of(u: int, v: int) -> int:
	return _group_of.get(mini(u, v) * n_cells + maxi(u, v), -1)


## Indices, within group g, of the pairs that hold colours a and b (none if a == b).
func swapping(col: PackedByteArray, g: int, a: int, b: int) -> PackedInt32Array:
	var out := PackedInt32Array()
	if a == b or g < 0 or g >= n_groups:
		return out
	var pu := pairs_u[g]
	var pv := pairs_v[g]
	for k in pu.size():
		var cu := col[pu[k]]
		var cv := col[pv[k]]
		if (cu == a and cv == b) or (cu == b and cv == a):
			out.append(k)
	return out


## Applies the move in place; returns the cells that changed colour.
func move(col: PackedByteArray, g: int, a: int, b: int) -> PackedInt32Array:
	var changed := PackedInt32Array()
	var pu := pairs_u[g] if g >= 0 and g < n_groups else PackedInt32Array()
	for k in swapping(col, g, a, b):
		var u := pu[k]
		var v := pairs_v[g][k]
		var c := col[u]
		col[u] = col[v]
		col[v] = c
		changed.append(u)
		changed.append(v)
	return changed
