extends SceneTree
## Headless tests of the foam game against the Python twin (tools/export_godot.py).
## godot --headless --path game --script res://tests/run_tests.gd

var _fails := 0
var _checks := 0


func _init() -> void:
	var index: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://data/index.json"))
	for entry in index["levels"]:
		_level(entry["id"])
	_refusals()
	print("TESTS %d checks, %d failed: %s" % [_checks, _fails, "PASS" if _fails == 0 else "FAIL"])
	quit(0 if _fails == 0 else 1)


func _ok(cond: bool, what: String) -> void:
	_checks += 1
	if not cond:
		_fails += 1
		print("FAIL ", what)


func _level(id: String) -> void:
	var lv := FoamLevel.load_level(id)
	_ok(lv.error.is_empty(), "%s loads (%s)" % [id, lv.error])
	if not lv.error.is_empty():
		return
	var f := lv.foam
	# every edge in one group, no cell twice in a group, group_of in both orders
	var matching := true
	for g in f.n_groups:
		var seen := {}
		for k in f.pairs_u[g].size():
			for c in [f.pairs_u[g][k], f.pairs_v[g][k]]:
				matching = matching and not seen.has(c)
				seen[c] = true
	_ok(matching, id + ": groups are matchings")
	var both := true
	for e in f.edge_u.size():
		both = both and f.group_of(f.edge_u[e], f.edge_v[e]) == f.edge_group[e]
		both = both and f.group_of(f.edge_v[e], f.edge_u[e]) == f.edge_group[e]
	_ok(both and f.group_of(0, 0) == -1, id + ": group_of")
	# the raster names every cell, and the site of a cell lies in that cell or next to it
	var present := {}
	for k in range(0, lv.labels.size(), 2):
		present[lv.labels[k] | (lv.labels[k + 1] << 8)] = true
	_ok(present.size() == f.n_cells, id + ": every cell on the raster")
	_ok(lv.cell_at(Vector2(-0.1, 0.5)) == -1 and lv.cell_at(Vector2(0.5, 1.0)) == -1, id + ": outside")
	# every move is its own inverse and keeps the colour counts
	var n := lv.colours.size()
	var inverse := true
	for g in f.n_groups:
		for a in n:
			for b in range(a + 1, n):
				var c := lv.start.duplicate()
				var ch := f.move(c, g, a, b)
				inverse = inverse and (ch.is_empty() == (c == lv.start))
				inverse = inverse and _counts(c, n) == _counts(lv.start, n)
				f.move(c, g, a, b)
				inverse = inverse and c == lv.start
	_ok(inverse, id + ": moves undo themselves and keep the colour counts")
	_ok(f.move(lv.start.duplicate(), 0, 1, 1).is_empty(), id + ": a = b does nothing")
	# golden: the reversed scramble gives the Python colours at every move
	var gold: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/golden/%s.json" % id))
	var s := Session.new(lv)
	_ok(s.col == FoamLevel._bytes(gold["start"]) and not s.solved(), id + ": scrambled start")
	var same := true
	for i in gold["moves"].size():
		var m: Array = gold["moves"][i]
		same = same and s.play(Vector3i(int(m[0]), int(m[1]), int(m[2])))
		same = same and s.col == FoamLevel._bytes(gold["states"][i])
	_ok(same, id + ": golden states")
	_ok(s.solved() and s.moves() == lv.par, id + ": solved at par")
	# undo, redo, same move again, reset
	_ok(s.undo() and not s.solved() and s.moves() == lv.par - 1, id + ": undo")
	_ok(s.redo() and s.solved() and not s.redo(), id + ": redo")
	var last: Vector3i = s.history[-1]
	_ok(s.play(last) and s.moves() == lv.par - 1 and s.future.size() == 1, id + ": same move takes it back")
	s.reset()
	_ok(s.col == lv.start and s.moves() == 0 and not s.undo(), id + ": reset")
	# pointing: for every edge, its two cells give the edge's group
	var pointed := true
	for e in f.edge_u.size():
		var m2 := s.move_between(f.edge_u[e], f.edge_v[e])
		var differ := s.col[f.edge_u[e]] != s.col[f.edge_v[e]]
		pointed = pointed and ((m2.x == f.edge_group[e]) == differ) and (differ or m2.x == -1)
	_ok(pointed, id + ": move_between")


func _refusals() -> void:
	var f := Foam.new()
	var err := f.setup(1.0, [[0, 0], [1, 0], [0, 1]], [[0, 1], [1, 2]], [0, 0])
	_ok(not err.is_empty(), "two pairs of a group sharing a cell are refused")
	_ok(not FoamLevel.load_level("none").error.is_empty(), "a missing level reports an error")


func _counts(c: PackedByteArray, n: int) -> Array:
	var out := []
	out.resize(n)
	out.fill(0)
	for v in c:
		out[v] += 1
	return out
