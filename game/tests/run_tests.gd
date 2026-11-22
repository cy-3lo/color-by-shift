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
	_progress()
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
	# hints: from the start, following the hints solves at par; after a wrong move, the next
	# move is its undo and following the hints solves in par + 1
	var area := Hint.areas(lv)
	_ok(_follow(lv, s, area) == lv.par and s.solved(), id + ": hints solve at par")
	s.reset()
	var wrong := Vector3i(-1, 0, 0)
	for g in f.n_groups:
		for a in n:
			for b in range(a + 1, n):
				var m3 := Vector3i(g, a, b)
				if wrong.x >= 0 or m3 in lv.solution or not s.play(m3):
					continue
				if Hint.next_moves(lv, s.col, s.history) == [m3]:
					wrong = m3
				else:
					s.undo()
	_ok(wrong.x >= 0, id + ": a wrong move exists")
	_ok(_follow(lv, s, area) == lv.par + 1 and s.solved(), id + ": after a wrong move, hints undo it and solve")
	s.reset()
	# pointing: for every edge, its two cells give the edge's group, and a point of its side
	# gives these two cells (or a pair of the same side point shared with a third cell)
	var pointed := true
	var found := 0
	for e in f.edge_u.size():
		var m2 := s.move_between(f.edge_u[e], f.edge_v[e])
		var differ := s.col[f.edge_u[e]] != s.col[f.edge_v[e]]
		pointed = pointed and ((m2.x == f.edge_group[e]) == differ) and (differ or m2.x == -1)
		for q in lv.sides[e]:
			var pr := lv.pair_at(q)
			if mini(pr.x, pr.y) == f.edge_u[e] and maxi(pr.x, pr.y) == f.edge_v[e]:
				found += 1
				break
	_ok(pointed, id + ": move_between")
	var with_side := 0
	for e in f.edge_u.size():
		with_side += int(not lv.sides[e].is_empty())
	_ok(found == with_side, "%s: every edge with a side can be pointed at (%d of %d)" % [id, found, with_side])
	_ok(with_side >= 0.99 * f.edge_u.size(), "%s: at most 1 %% of the edges have no side (%d of %d)" % [id, f.edge_u.size() - with_side, f.edge_u.size()])


## Plays the first next move of the hint until solved; the number of moves played (-1 if a hint
## cell is missing or the hints loop).
func _follow(lv: FoamLevel, s: Session, area: PackedFloat32Array) -> int:
	var steps := 0
	while not s.solved():
		var nm := Hint.next_moves(lv, s.col, s.history)
		if nm.is_empty() or Hint.cell(lv, s.col, s.history, area) < 0 or steps > 3 * lv.par + 4:
			return -1
		s.play(nm[0])
		steps += 1
	return steps


## Levels open one by one until the last level of par 2 or less is solved, then all open;
## progress survives a save and a load.
func _progress() -> void:
	var ids: Array[String] = ["A", "B", "C", "D", "E", "F"]
	var pars := [1, 1, 2, 2, 3, 4]
	var path := "user://test_progress.json"
	var pr := Progress.new(ids, pars, 2, path)
	pr.solved.clear()
	_ok(pr.open_all_after == 3, "progress: level that opens all")
	_ok(pr.is_open(0) and not pr.is_open(1) and pr.start() == 0, "progress: only the first open")
	pr.mark_solved(0)
	_ok(pr.is_open(1) and not pr.is_open(2) and pr.start() == 1, "progress: solving opens the next")
	pr.mark_solved(1)
	pr.mark_solved(2)
	_ok(pr.is_open(3) and not pr.is_open(4) and not pr.is_open(5), "progress: still one by one")
	pr.mark_solved(3)
	_ok(pr.is_open(4) and pr.is_open(5), "progress: all open after the last par-2 level")
	pr.set_current(5)
	var back := Progress.new(ids, pars, 2, path)
	_ok(back.solved.size() == 4 and back.start() == 5 and back.is_open(5), "progress: saved and loaded")
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	var fresh := Progress.new(ids, pars, 2, "")
	fresh.current = "F"
	_ok(fresh.start() == 0, "progress: a closed current level is not a start")


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
