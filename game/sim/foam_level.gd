class_name FoamLevel
extends RefCounted
## A level written by tools/export_godot.py: a foam, the picture to recover, the scrambled
## start, and the raster that gives the cell of every pixel.

var id := ""
var foam: Foam
var colours: Array[Color] = []
var target := PackedByteArray()
var start := PackedByteArray()
var par := 0
var solution: Array[Vector3i] = []  ## one shortest way from start to target (the reversed scramble)
var labels := PackedByteArray()  ## uint16 little-endian per pixel, row by row
var label_w := 0
var label_h := 0
var sides: Array[PackedVector2Array] = []  ## per edge, points on the shared side (foam coordinates)
var shapes: Array[PackedVector2Array] = []  ## per cell, its outline (foam coordinates), clockwise
var painting := ""  ## res:// path of the reward scan, "" if none
var artist := ""
var date := ""
var title := ""
var error := ""


static func load_level(level_id: String, folder := "res://data") -> FoamLevel:
	var lv := FoamLevel.new()
	lv.id = level_id
	var path := "%s/levels/%s.json" % [folder, level_id]
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		lv.error = "cannot read " + path
		return lv
	var d: Variant = JSON.parse_string(text)
	if typeof(d) != TYPE_DICTIONARY:
		lv.error = "not a level: " + path
		return lv
	lv.foam = Foam.new()
	lv.error = lv.foam.setup(float(d["aspect"]), d["sites"], d["edges"], d["group"])
	if not lv.error.is_empty():
		return lv
	for s in d["sides"]:
		var pts := PackedVector2Array()
		for k in range(0, s.size(), 2):
			pts.append(Vector2(s[k], s[k + 1]))
		lv.sides.append(pts)
	if lv.sides.size() != lv.foam.edge_u.size():
		lv.error = "sides and edges differ in number"
		return lv
	for c in d["colours"]:
		lv.colours.append(Color.html(c))
	lv.target = _bytes(d["target"])
	lv.start = _bytes(d["start"])
	lv.par = int(d["par"])
	for m in d.get("solution", []):
		lv.solution.append(Vector3i(int(m[0]), int(m[1]), int(m[2])))
	lv.artist = d.get("artist", "")
	lv.date = d.get("date", "")
	lv.title = d.get("title", "")
	if not String(d.get("painting", "")).is_empty():
		lv.painting = "%s/paintings/%s" % [folder, d["painting"]]
	var n := lv.foam.n_cells
	if lv.target.size() != n or lv.start.size() != n:
		lv.error = "picture does not fit the foam"
		return lv
	for i in n:
		if lv.target[i] >= lv.colours.size() or lv.start[i] >= lv.colours.size():
			lv.error = "colour outside the palette"
			return lv
	lv.label_w = int(d["labels"]["width"])
	lv.label_h = int(d["labels"]["height"])
	var packed := FileAccess.get_file_as_bytes("%s/levels/%s" % [folder, d["labels"]["file"]])
	lv.labels = packed.decompress(lv.label_w * lv.label_h * 2, FileAccess.COMPRESSION_GZIP)
	if lv.labels.size() != lv.label_w * lv.label_h * 2:
		lv.error = "raster of cells missing or of the wrong size"
		return lv
	var k := int(d["shapes"]["points"])
	var raw := FileAccess.get_file_as_bytes("%s/levels/%s" % [folder, d["shapes"]["file"]])
	raw = raw.decompress(n * k * 4, FileAccess.COMPRESSION_GZIP)
	if raw.size() != n * k * 4:
		lv.error = "outlines of the cells missing or of the wrong size"
		return lv
	var scale := Vector2(lv.foam.aspect, 1.0) / 65535.0
	for c in n:
		var pts := PackedVector2Array()
		pts.resize(k)
		for j in k:
			var o := (c * k + j) * 4
			pts[j] = Vector2(raw.decode_u16(o), raw.decode_u16(o + 2)) * scale
		lv.shapes.append(pts)
	return lv


## Cell at a point of the picture given in [0, 1] x [0, 1]; -1 outside.
func cell_at(uv: Vector2) -> int:
	if uv.x < 0.0 or uv.y < 0.0 or uv.x >= 1.0 or uv.y >= 1.0:
		return -1
	var k := (int(uv.y * label_h) * label_w + int(uv.x * label_w)) * 2
	return labels[k] | (labels[k + 1] << 8)


## The cell at a point (foam coordinates) and the neighbour whose shared side is nearest to
## it; (-1, -1) outside the picture.
func pair_at(p: Vector2) -> Vector2i:
	var u := cell_at(p / Vector2(foam.aspect, 1.0))
	if u < 0:
		return Vector2i(-1, -1)
	var v := -1
	var best := INF
	for w in foam.neighbours[u]:
		for q in sides[foam.edge_of(u, w)]:
			var d := p.distance_squared_to(q)
			if d < best:
				best = d
				v = w
	return Vector2i(u, v) if v >= 0 else Vector2i(-1, -1)


static func _bytes(a: Array) -> PackedByteArray:
	var out := PackedByteArray()
	out.resize(a.size())
	for i in a.size():
		out[i] = int(a[i])
	return out
