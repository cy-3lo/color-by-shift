class_name Icons
extends RefCounted
## Interface icons in several sets, drawn from SVG strings (24 x 24) at the size asked for:
## the game's own sets below, and sets fetched from free icon libraries (data/icons.json).
## Every set has the four icons of the bar: next (level), reset, undo, redo.

const HEAD := "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24' fill='white' stroke='none'>"
const THIN := "<g fill='none' stroke='white' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
const THICK := "<g fill='none' stroke='white' stroke-width='3.4' stroke-linecap='round' stroke-linejoin='round'>"
const BAR := " fill='none' stroke='white' stroke-width='3.2'"
const MIRROR := "<g transform='translate(24 0) scale(-1 1)'>"
# shapes shared by several sets
const U_TURN := "<polyline points='8.5,5 3.5,10 8.5,15'/><path d='M3.5 10 H14.5 a5 5 0 0 1 0 10 H10'/>"
const ROUND := "<path d='M19 12 a7 7 0 1 1 -2.05 -4.95'/><polyline points='17.2,2.8 17.2,7.3 12.7,7.3'/>"
const U_SOLID := "<path d='M7 9 H14.5 a5.5 5.5 0 0 1 0 11 H9'" + BAR + "/><polygon points='8.5,3 8.5,15 1.5,9'/>"
const ROUND_SOLID := "<path d='M18.4 12 a6.4 6.4 0 1 1 -3.2 -5.54'" + BAR + "/><polygon points='20.2,9.3 17.6,2.3 12.8,8.6'/>"

const ORDER: Array[String] = ["solid", "line", "bold", "media"]
const SETS := {
	"solid": {
		"next": "<polygon points='6.5,4 6.5,20 19,12'/>",
		"reset": ROUND_SOLID,
		"undo": U_SOLID,
		"redo": MIRROR + U_SOLID + "</g>",
	},
	"line": {
		"next": THIN + "<polyline points='9.5,5 16.5,12 9.5,19'/></g>",
		"reset": THIN + ROUND + "</g>",
		"undo": THIN + U_TURN + "</g>",
		"redo": MIRROR + THIN + U_TURN + "</g></g>",
	},
	"bold": {
		"next": THICK + "<path d='M4 12 H19'/><polyline points='13,6 19,12 13,18'/></g>",
		"reset": THICK + ROUND + "</g>",
		"undo": THICK + U_TURN + "</g>",
		"redo": MIRROR + THICK + U_TURN + "</g></g>",
	},
	"media": {
		"next": "<polygon points='4.5,5 4.5,19 15,12'/><rect x='16.3' y='5' width='3.2' height='14'/>",
		"reset": "<rect x='4.5' y='5' width='3.2' height='14'/><polygon points='19.5,5 19.5,19 9,12'/>",
		"undo": MIRROR + ROUND_SOLID + "</g>",
		"redo": ROUND_SOLID,
	},
}

static var _cache := {}
static var _fetched := {}  ## sets of game/data/icons.json (tools/fetch_icons.py): name -> {"icons": {...}}
static var _loaded := false


static func _load() -> void:
	if _loaded:
		return
	_loaded = true
	var d: Variant = JSON.parse_string(FileAccess.get_file_as_string("res://data/icons.json"))
	if typeof(d) == TYPE_DICTIONARY:
		_fetched = d


## Every set: the game's own, then the fetched libraries.
static func names() -> Array[String]:
	_load()
	var out: Array[String] = ORDER.duplicate()
	for k in _fetched:
		out.append(k)
	return out


## White icon `name` of the set `kind` as a texture `px` pixels wide; tint it when drawing.
static func texture(kind: String, name: String, px: int) -> Texture2D:
	_load()
	var key := "%s/%s@%d" % [kind, name, px]
	if not _cache.has(key):
		var svg: String
		if _fetched.has(kind):
			svg = _fetched[kind]["icons"][name]
		else:
			var set_: Dictionary = SETS.get(kind, SETS[ORDER[0]])
			svg = HEAD + String(set_[name]) + "</svg>"
		var img := Image.new()
		if img.load_svg_from_string(svg, px / 24.0) != OK:
			img = Image.create(px, px, false, Image.FORMAT_RGBA8)
		_cache[key] = ImageTexture.create_from_image(img)
	return _cache[key]


## The set after `kind`.
static func after(kind: String) -> String:
	var all := names()
	return all[(all.find(kind) + 1) % all.size()]
