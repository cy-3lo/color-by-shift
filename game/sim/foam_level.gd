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
var labels := PackedByteArray()  ## uint16 little-endian per pixel, row by row
var label_w := 0
var label_h := 0
var painting := ""  ## res:// path of the reward scan, "" if none
var artist := ""
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
	for c in d["colours"]:
		lv.colours.append(Color.html(c))
	lv.target = _bytes(d["target"])
	lv.start = _bytes(d["start"])
	lv.par = int(d["par"])
	lv.artist = d.get("artist", "")
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


## Cell at a point of the picture given in [0, 1] x [0, 1]; -1 outside.
func cell_at(uv: Vector2) -> int:
	if uv.x < 0.0 or uv.y < 0.0 or uv.x >= 1.0 or uv.y >= 1.0:
		return -1
	var k := (int(uv.y * label_h) * label_w + int(uv.x * label_w)) * 2
	return labels[k] | (labels[k + 1] << 8)


static func _bytes(a: Array) -> PackedByteArray:
	var out := PackedByteArray()
	out.resize(a.size())
	for i in a.size():
		out[i] = int(a[i])
	return out
