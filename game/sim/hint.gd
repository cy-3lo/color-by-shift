class_name Hint
extends RefCounted
## The hint: the cell to point at after a pause.
##
## Next moves: if some order of distinct moves of the level's solution leads from the present
## colours to the picture, the first moves of the shortest such orders (one move, or several
## when they commute); otherwise the player has strayed and the next move is the undo of the
## latest move. The hint cell is the largest misplaced cell among the cells these moves swap
## (the largest swapped cell if none of them is misplaced).


## The next moves from colours col, after the moves in history.
static func next_moves(lv: FoamLevel, col: PackedByteArray, history: Array[Vector3i]) -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	if col == lv.target:
		return out
	var c := col.duplicate()
	var used := []
	used.resize(lv.solution.size())
	used.fill(false)
	for depth in range(1, lv.solution.size() + 1):
		_search(lv, c, used, depth, -1, out)
		if not out.is_empty():
			return out
	if not history.is_empty():
		out.append(history[-1])
	return out


## Depth-first over orders of unused solution moves; adds to out the first move of every order
## of exactly `left` moves that reaches the picture. `first` is the first move of this branch.
static func _search(lv: FoamLevel, c: PackedByteArray, used: Array, left: int, first: int, out: Array[Vector3i]) -> void:
	if left == 0:
		if c == lv.target and not out.has(lv.solution[first]):
			out.append(lv.solution[first])
		return
	for i in lv.solution.size():
		if used[i]:
			continue
		var m := lv.solution[i]
		if lv.foam.move(c, m.x, m.y, m.z).is_empty():
			continue  # changes nothing here: never on a shortest way
		used[i] = true
		_search(lv, c, used, left - 1, i if first < 0 else first, out)
		used[i] = false
		lv.foam.move(c, m.x, m.y, m.z)  # every move is its own inverse


## The cell to point at, or -1. area: the area of every cell.
static func cell(lv: FoamLevel, col: PackedByteArray, history: Array[Vector3i], area: PackedFloat32Array) -> int:
	var best := -1
	var best_any := -1
	for m in next_moves(lv, col, history):
		var pu: PackedInt32Array = lv.foam.pairs_u[m.x]
		var pv: PackedInt32Array = lv.foam.pairs_v[m.x]
		for k in lv.foam.swapping(col, m.x, m.y, m.z):
			for c in [pu[k], pv[k]]:
				if best_any < 0 or area[c] > area[best_any]:
					best_any = c
				if col[c] != lv.target[c] and (best < 0 or area[c] > area[best]):
					best = c
	return best if best >= 0 else best_any


## The area of every cell (shoelace formula on its outline), in foam units.
static func areas(lv: FoamLevel) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(lv.foam.n_cells)
	for c in lv.foam.n_cells:
		var pts := lv.shapes[c]
		var a := 0.0
		for j in pts.size():
			var q0 := pts[j]
			var q1 := pts[(j + 1) % pts.size()]
			a += q0.x * q1.y - q1.x * q0.y
		out[c] = absf(a) * 0.5
	return out
