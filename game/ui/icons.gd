class_name Icons
extends RefCounted
## Interface icons, drawn from SVG strings (24 x 24, solid shapes) at the size asked for.

const HEAD := "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24' fill='white' stroke='none'>"
const BOLD := " fill='none' stroke='white' stroke-width='3.2' stroke-linecap='butt' stroke-linejoin='round'"
const SHAPES := {
	"undo": "<path d='M7 9 H14.5 a5.5 5.5 0 0 1 0 11 H9'%s/><polygon points='8.5,3 8.5,15 1.5,9'/>",
	"redo": "<path d='M17 9 H9.5 a5.5 5.5 0 0 0 0 11 H15'%s/><polygon points='15.5,3 15.5,15 22.5,9'/>",
	"reset": "<path d='M18.4 12 a6.4 6.4 0 1 1 -3.2 -5.54'%s/><polygon points='20.2,9.3 17.6,2.3 12.8,8.6'/>",
	"prev": "<polygon points='17,4 17,20 5.5,12'/>",
	"next": "<polygon points='7,4 7,20 18.5,12'/>",
	"forward": "<polygon points='5,4 5,20 15.5,12'/><rect x='16.5' y='4' width='3.2' height='16'/>",
}

static var _cache := {}


## White icon `name` as a texture `px` pixels wide; tint it when drawing.
static func texture(name: String, px: int) -> Texture2D:
	var key := "%s@%d" % [name, px]
	if not _cache.has(key):
		var img := Image.new()
		var err := img.load_svg_from_string(HEAD + String(SHAPES[name]).replace("%s", BOLD) + "</svg>", px / 24.0)
		if err != OK:
			img = Image.create(px, px, false, Image.FORMAT_RGBA8)
		_cache[key] = ImageTexture.create_from_image(img)
	return _cache[key]
