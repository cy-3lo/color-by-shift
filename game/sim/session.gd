class_name Session
extends RefCounted
## One attempt at a level: current colours, the moves played, undo and redo.
## A move is a Vector3i (group, a, b) with a < b.

var level: FoamLevel
var col := PackedByteArray()
var history: Array[Vector3i] = []
var future: Array[Vector3i] = []
var last_changed := PackedInt32Array()  ## cells changed by the latest move, undo or redo


func _init(p_level: FoamLevel) -> void:
	level = p_level
	reset()


func reset() -> void:
	col = level.start.duplicate()
	history.clear()
	future.clear()
	last_changed = PackedInt32Array()


func moves() -> int:
	return history.size()


func solved() -> bool:
	return col == level.target


## The move a click between cells u and v would play; (-1, 0, 0) if none.
func move_between(u: int, v: int) -> Vector3i:
	var g := level.foam.group_of(u, v)
	if g < 0 or col[u] == col[v]:
		return Vector3i(-1, 0, 0)
	return Vector3i(g, mini(col[u], col[v]), maxi(col[u], col[v]))


## Plays a move. The same move as the latest one takes it back. Returns false if nothing changed.
func play(m: Vector3i) -> bool:
	if m.x < 0 or level.foam.swapping(col, m.x, m.y, m.z).is_empty():
		return false
	if not history.is_empty() and history[-1] == m:
		return undo()
	last_changed = level.foam.move(col, m.x, m.y, m.z)
	history.append(m)
	future.clear()
	return true


func undo() -> bool:
	if history.is_empty():
		return false
	var m: Vector3i = history.pop_back()
	last_changed = level.foam.move(col, m.x, m.y, m.z)
	future.append(m)
	return true


func redo() -> bool:
	if future.is_empty():
		return false
	var m: Vector3i = future.pop_back()
	last_changed = level.foam.move(col, m.x, m.y, m.z)
	history.append(m)
	return true
