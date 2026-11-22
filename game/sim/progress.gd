class_name Progress
extends RefCounted
## Which levels are solved, which is current, which are open.
##
## Levels open one by one: the first is open, and each solved level opens the next. Once the
## last level of par open_all_par or less is solved, every level is open. Kept as JSON
## ({"solved": [ids], "current": id}); nothing is written when path is "".

var ids: Array[String] = []
var solved := {}  ## id -> true
var current := ""
var path := ""
var open_all_after := -1  ## index of the level whose solving opens every level, -1 for none
var all_open := false  ## every level open regardless (the headless drag test)


## pars: the par of every level, in the order of ids.
func _init(p_ids: Array[String], pars: Array, open_all_par: int, p_path := "") -> void:
	ids = p_ids
	path = p_path
	for i in ids.size():
		if int(pars[i]) <= open_all_par:
			open_all_after = i
	if path.is_empty() or not FileAccess.file_exists(path):
		return
	var d: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(d) != TYPE_DICTIONARY:
		return
	for id in d.get("solved", []):
		if String(id) in ids:
			solved[String(id)] = true
	current = String(d.get("current", ""))


func save() -> void:
	if path.is_empty():
		return
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify({"solved": solved.keys(), "current": current}))


func is_solved(i: int) -> bool:
	return i >= 0 and i < ids.size() and solved.has(ids[i])


func is_open(i: int) -> bool:
	if i < 0 or i >= ids.size():
		return false
	return all_open or i == 0 or is_solved(i) or is_solved(i - 1) or is_solved(open_all_after)


func mark_solved(i: int) -> void:
	if not is_solved(i):
		solved[ids[i]] = true
		save()


func set_current(i: int) -> void:
	if ids[i] != current:
		current = ids[i]
		save()


## Where to start: the current level if open, else the first open level not solved, else 0.
func start() -> int:
	var i := ids.find(current)
	if i >= 0 and is_open(i):
		return i
	for k in ids.size():
		if is_open(k) and not is_solved(k):
			return k
	return 0
