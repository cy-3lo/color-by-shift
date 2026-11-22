class_name Icons
extends RefCounted
## Interface icons from Ionicons (MIT, data/icons.json, fetched by tools/fetch_icons.py), drawn
## from their SVG strings (24 x 24, white) at the size asked for: next, reset, undo, redo,
## credits, settings, lock.

const SET := "ionicons"

static var _cache := {}
static var _icons := {}  ## name -> svg
static var _loaded := false


static func _load() -> void:
	if _loaded:
		return
	_loaded = true
	var d: Variant = JSON.parse_string(FileAccess.get_file_as_string("res://data/icons.json"))
	if typeof(d) == TYPE_DICTIONARY and d.has(SET):
		_icons = d[SET]["icons"]


## White icon `name` as a texture `px` pixels wide (empty if unknown); tint it when drawing.
static func texture(name: String, px: int) -> Texture2D:
	_load()
	var key := "%s@%d" % [name, px]
	if not _cache.has(key):
		var img := Image.new()
		if img.load_svg_from_string(String(_icons.get(name, "")), px / 24.0) != OK:
			img = Image.create(px, px, false, Image.FORMAT_RGBA8)
		_cache[key] = ImageTexture.create_from_image(img)
	return _cache[key]
