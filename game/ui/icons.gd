class_name Icons
extends RefCounted
## Interface icons, drawn from SVG strings (24 x 24, round strokes) at the size asked for.

const HEAD := "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24' fill='none' stroke='white' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
const SHAPES := {
	"undo": "<polyline points='8.5,5 3.5,10 8.5,15'/><path d='M3.5 10 H14.5 a5 5 0 0 1 0 10 H10'/>",
	"redo": "<polyline points='15.5,5 20.5,10 15.5,15'/><path d='M20.5 10 H9.5 a5 5 0 0 0 0 10 H14'/>",
	"reset": "<path d='M19 12 a7 7 0 1 1 -2.05 -4.95'/><polyline points='17.2,2.8 17.2,7.3 12.7,7.3'/>",
	"prev": "<polyline points='14.5,5 7.5,12 14.5,19'/>",
	"next": "<polyline points='9.5,5 16.5,12 9.5,19'/>",
	"forward": "<path d='M4 12 H19'/><polyline points='13,6 19,12 13,18'/>",
}

static var _cache := {}


## White icon `name` as a texture `px` pixels wide; tint it when drawing.
static func texture(name: String, px: int) -> Texture2D:
	var key := "%s@%d" % [name, px]
	if not _cache.has(key):
		var img := Image.new()
		var err := img.load_svg_from_string(HEAD + SHAPES[name] + "</svg>", px / 24.0)
		if err != OK:
			img = Image.create(px, px, false, Image.FORMAT_RGBA8)
		_cache[key] = ImageTexture.create_from_image(img)
	return _cache[key]
