extends Control
## Play screen: the foam picture and a strip of icons (a bar at the bottom when the window
## is taller than wide, as on a phone).
## The gesture: press a cell and drag toward a neighbour, with the mouse or a finger. Every
## pair of the move follows the drag (each cell travels to its partner's place and takes its
## shape, the colour that was picked passing above); below drag.commit of the way it is a preview, beyond it
## the picture tips into the shifted state (and comes back if the drag returns below
## drag.release before letting go).
## Options after "--": --level=ID, --solved (shows the level recovered, to look at the
## reward), --smoke (headless: solve every level by drags). In a browser:
## index.html?level=ID&solved.

const SHADER := preload("res://scenes/picture.gdshader")

enum Phase { IDLE, DRAG, SNAP, BACK }

var cfg: Dictionary
var ids: Array[String] = []
var at := 0
var level: FoamLevel
var session: Session

var rect := Rect2()  ## the picture on screen
var strip := Rect2()  ## the strip (right) or the bar (bottom)
var boxes := {}  ## "number", "par", "info" -> Rect2: the boxes of the bar above the buttons
var icon_set := "solid"
var note := ""  ## a word shown for a moment (the icon set just chosen)
var note_until := -1.0
var unit := 1.0  ## scale of the interface (larger when the window is taller than wide)
var portrait := false
var site_px := PackedVector2Array()
var shape_px: Array[PackedVector2Array] = []  ## outline of every cell, pixels from the picture's corner
var centre_px := PackedVector2Array()  ## middle of every outline
var moving := PackedInt32Array()  ## cells of the pairs of the move under way: u, v, u, v ...
var top_colour := -1  ## the colour that passes above the other
var eased := 0.0  ## progress of the move as drawn
var buttons := {}  ## name -> Rect2

var picture: ColorRect
var tiles: Control  ## the cells of the move under way, drawn above the picture
var overlay: Control
var material_: ShaderMaterial
var cell_image: Image
var cell_texture: ImageTexture

var phase := Phase.IDLE
var shift := 0.0  ## progress of the move under way, 0..1
var pending := Vector3i(-1, 0, 0)  ## the move under way
var action := ""  ## what to do with it once shifted: "play", "undo" or "redo"
var press_pos := Vector2.ZERO
var drag_u := -1
var drag_v := -1
var drag_frac := 0.0
var committed := false
var solved_at := -1.0
var now := 0.0
var preview := false  ## start from the recovered picture


func _ready() -> void:
	cfg = JSON.parse_string(FileAccess.get_file_as_string("res://data/config.json"))
	var index: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://data/index.json"))
	for e in index["levels"]:
		ids.append(e["id"])
	var want: String = cfg.get("start_level", "")
	icon_set = cfg.get("icons", "solid")
	var smoke := false
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--level="):
			want = a.get_slice("=", 1)
		smoke = smoke or a == "--smoke"
		preview = preview or a == "--solved"
	if OS.has_feature("web"):
		var query := str(JavaScriptBridge.eval("window.location.search", true))
		for part in query.trim_prefix("?").split("&"):
			if part.begins_with("level="):
				want = part.get_slice("=", 1)
			preview = preview or part == "solved"
	at = maxi(ids.find(want), 0)

	set_anchors_preset(Control.PRESET_FULL_RECT)
	var bg := ColorRect.new()
	bg.color = _c("background")
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)
	picture = ColorRect.new()
	picture.mouse_filter = Control.MOUSE_FILTER_IGNORE
	material_ = ShaderMaterial.new()
	material_.shader = SHADER
	picture.material = material_
	add_child(picture)
	tiles = Control.new()
	tiles.mouse_filter = Control.MOUSE_FILTER_IGNORE
	tiles.clip_contents = true
	tiles.draw.connect(_draw_tiles)
	add_child(tiles)
	overlay = Control.new()
	overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
	overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	overlay.draw.connect(_draw_overlay)
	add_child(overlay)
	get_viewport().size_changed.connect(_layout)
	_load(at)
	if smoke:
		_smoke()


func _c(key: String) -> Color:
	return Color.html(cfg["colours"][key])


# ---- level --------------------------------------------------------------------------------
func _load(k: int) -> void:
	at = posmod(k, ids.size())
	level = FoamLevel.load_level(ids[at])
	if not level.error.is_empty():
		push_error("%s: %s" % [ids[at], level.error])
		return
	session = Session.new(level)
	var n := level.foam.n_cells
	var labels := Image.create_from_data(
		level.label_w, level.label_h, false, Image.FORMAT_RG8, level.labels)
	cell_image = Image.create(n, 3, false, Image.FORMAT_RGBA8)
	for i in n:
		var t := level.colours[level.target[i]]
		cell_image.set_pixel(i, 1, Color(t.r, t.g, t.b, level.target[i] / 255.0))
	cell_texture = ImageTexture.create_from_image(cell_image)
	material_.set_shader_parameter("labels", ImageTexture.create_from_image(labels))
	material_.set_shader_parameter("cells", cell_texture)
	material_.set_shader_parameter("label_size", Vector2(level.label_w, level.label_h))
	var line := _c("line")
	material_.set_shader_parameter("line_colour", Vector3(line.r, line.g, line.b))
	var hole := _c("hole")
	material_.set_shader_parameter("hole_colour", Vector3(hole.r, hole.g, hole.b))
	if not level.painting.is_empty() and ResourceLoader.exists(level.painting):
		material_.set_shader_parameter("painting", load(level.painting))
	phase = Phase.IDLE
	shift = 0.0
	pending = Vector3i(-1, 0, 0)
	solved_at = -1.0
	if preview:
		session.col = level.target.duplicate()
		solved_at = now
	_upload()
	_layout()


func _layout() -> void:
	if level == null or level.foam == null:
		return
	var size_ := get_viewport_rect().size
	portrait = size_.y > size_.x
	# the base window is 1280 x 720; a tall window keeps 1280 across, so its icons grow
	unit = clampf(1.4 * size_.y / size_.x, 1.0, 3.2) if portrait else 1.0
	var m := float(cfg["picture"]["margin"])
	var room: Rect2
	if portrait:
		var bh := float(cfg["strip"]["bar_height"]) * unit
		strip = Rect2(0.0, size_.y - bh, size_.x, bh)
		room = Rect2(0.0, 0.0, size_.x, size_.y - bh)
	else:
		var sw := float(cfg["strip"]["width"])
		strip = Rect2(size_.x - sw, 0.0, sw, size_.y)
		room = Rect2(0.0, 0.0, size_.x - sw, size_.y)
	var w := minf(room.size.x - 2.0 * m, (room.size.y - 2.0 * m) * level.foam.aspect)
	if portrait:  # leave room under the picture for the count and the messages
		w = minf(w, (room.size.y - 2.0 * m - 170.0 * unit) * level.foam.aspect)
	var h := w / level.foam.aspect
	var top := m if portrait else (room.size.y - h) * 0.5
	rect = Rect2(Vector2((room.size.x - w) * 0.5, top).round(), Vector2(w, h))
	picture.position = rect.position
	picture.size = rect.size
	tiles.position = rect.position
	tiles.size = rect.size
	var to_px := rect.size / Vector2(level.foam.aspect, 1.0)
	shape_px.clear()
	centre_px.resize(level.foam.n_cells)
	for c in level.foam.n_cells:
		var pts := PackedVector2Array()
		var sum := Vector2.ZERO
		for q in level.shapes[c]:
			pts.append(q * to_px)
			sum += q * to_px
		shape_px.append(pts)
		centre_px[c] = sum / pts.size()
	material_.set_shader_parameter("picture_px", rect.size)
	material_.set_shader_parameter("line_px", float(cfg["picture"]["line_px"]) * unit)
	material_.set_shader_parameter("outline_px", float(cfg["picture"]["outline_px"]) * unit)
	site_px.resize(level.foam.n_cells)
	for i in level.foam.n_cells:
		site_px[i] = rect.position + level.foam.sites[i] / Vector2(level.foam.aspect, 1.0) * rect.size
	# one style for everything in the bar: boxes of the same width, the same gap between them
	var pad := float(cfg["strip"]["pad"]) * unit
	var names := ["next", "reset", "undo", "redo"]
	buttons = {}
	boxes = {}
	if portrait:  # boxes in a row under the picture, four square buttons in the bar
		var side := minf(strip.size.y - 2.0 * pad, (strip.size.x - pad * (names.size() + 1)) / names.size())
		var gap := (strip.size.x - side * names.size()) / (names.size() + 1)
		for k in names.size():
			buttons[names[k]] = Rect2(gap + k * (side + gap), strip.get_center().y - side * 0.5, side, side)
		var y0 := rect.end.y + pad
		var hh := strip.position.y - y0 - pad
		boxes["number"] = Rect2(gap, y0, side, hh * 0.5 - pad * 0.5)
		boxes["par"] = Rect2(gap, y0 + hh * 0.5 + pad * 0.5, side, hh * 0.5 - pad * 0.5)
		boxes["info"] = Rect2(gap + side + gap, y0, size_.x - 3.0 * gap - side, hh)
	else:  # top to bottom: number, par, painting, then the four square buttons
		var x := strip.position.x + pad
		var side := strip.size.x - 2.0 * pad
		for k in names.size():
			buttons[names[k]] = Rect2(x, size_.y - (names.size() - k) * (side + pad), side, side)
		boxes["number"] = Rect2(x, pad, side, float(cfg["strip"]["number_height"]))
		boxes["par"] = Rect2(x, boxes["number"].end.y + pad, side, float(cfg["strip"]["par_height"]))
		var y1: float = boxes["par"].end.y + pad
		boxes["info"] = Rect2(x, y1, side, buttons["next"].position.y - pad - y1)
	overlay.queue_redraw()


## Writes the colours of the cells (row 0) and marks the cells of the move under way (row 2).
func _upload() -> void:
	var n := level.foam.n_cells
	for i in n:
		cell_image.set_pixel(i, 0, level.colours[session.col[i]])
		cell_image.set_pixel(i, 2, Color(0, 0, 0, 1))
	moving = PackedInt32Array()
	if pending.x >= 0:
		var foam := level.foam
		for k in foam.swapping(session.col, pending.x, pending.y, pending.z):
			var u := foam.pairs_u[pending.x][k]
			var v := foam.pairs_v[pending.x][k]
			moving.append(u)
			moving.append(v)
			cell_image.set_pixel(u, 2, Color(0, 0, 1, 1))
			cell_image.set_pixel(v, 2, Color(0, 0, 1, 1))
	cell_texture.update(cell_image)
	tiles.queue_redraw()


# ---- the move under way -------------------------------------------------------------------
func _set_pending(m: Vector3i, what: String, above: int = -1) -> void:
	pending = m
	action = what
	top_colour = above if above >= 0 else m.z
	shift = 0.0
	_upload()


## Ends the move under way at once: applied if it was tipping over, dropped otherwise.
func _settle() -> void:
	if phase == Phase.SNAP:
		_apply()
	elif phase != Phase.IDLE:
		_drop()


func _apply() -> void:
	match action:
		"play":
			session.play(pending)
		"undo":
			session.undo()
		"redo":
			session.redo()
	_drop()
	solved_at = now if session.solved() else -1.0


func _drop() -> void:
	pending = Vector3i(-1, 0, 0)
	phase = Phase.IDLE
	shift = 0.0
	drag_u = -1
	drag_v = -1
	committed = false
	_upload()
	overlay.queue_redraw()


## Shifts a whole move without a drag (undo, redo).
func _run(m: Vector3i, what: String) -> void:
	_settle()
	_set_pending(m, what)
	phase = Phase.SNAP


# ---- pointer: press, drag, release --------------------------------------------------------
func press(pos: Vector2) -> void:
	for name_ in buttons:
		if buttons[name_].has_point(pos):
			_press_button(name_)
			return
	if solved_at >= 0.0:
		return
	_settle()
	var u := level.cell_at((pos - rect.position) / rect.size)
	if u < 0:
		return
	phase = Phase.DRAG
	press_pos = pos
	drag_u = u
	drag_v = -1
	drag_frac = 0.0
	committed = false


## Middle of the side shared by cells u and w on screen (the site of w if the side is unknown).
func _side_mid(u: int, w: int) -> Vector2:
	var pts := level.sides[level.foam.edge_of(u, w)]
	if pts.is_empty():
		return site_px[w]
	var mid := Vector2.ZERO
	for q in pts:
		mid += q
	return rect.position + mid / pts.size() / Vector2(level.foam.aspect, 1.0) * rect.size


## The neighbour of the pressed cell whose shared side lies most nearly in the direction of
## the drag, seen from the point pressed.
func _toward(dir: Vector2) -> int:
	var best := -2.0
	var v := -1
	for w in level.foam.neighbours[drag_u]:
		var c := dir.dot((_side_mid(drag_u, w) - press_pos).normalized())
		if c > best:
			best = c
			v = w
	return v


func drag(pos: Vector2) -> void:
	if phase != Phase.DRAG:
		return
	var d := pos - press_pos
	var dc: Dictionary = cfg["drag"]
	if d.length() < float(dc["dead_px"]) * unit:
		drag_frac = 0.0
		return
	if drag_v < 0 or (not committed and drag_frac < float(dc["rechoose"])):
		var v := _toward(d.normalized())
		if v != drag_v:
			drag_v = v
			_set_pending(session.move_between(drag_u, v), "play", session.col[drag_u])
	if drag_v < 0:
		return
	var span := site_px[drag_v] - site_px[drag_u]
	var travel := maxf(span.length(), float(dc["min_travel_px"]) * unit)
	drag_frac = clampf(d.dot(span.normalized()) / travel, 0.0, 1.0)
	if pending.x < 0:
		return
	if drag_frac >= float(dc["commit"]):
		committed = true
	elif drag_frac < float(dc["release"]):
		committed = false


func release() -> void:
	if phase != Phase.DRAG:
		return
	if pending.x >= 0 and committed:
		phase = Phase.SNAP
	elif pending.x >= 0:
		phase = Phase.BACK
	else:
		_drop()


func _press_button(name_: String) -> void:
	match name_:
		"prev":
			_load(at - 1)
		"next":
			_load(at + 1)
		"undo":
			_settle()
			if not session.history.is_empty():
				_run(session.history[-1], "undo")
		"redo":
			_settle()
			if not session.future.is_empty():
				_run(session.future[-1], "redo")
		"reset":
			_settle()
			session.reset()
			solved_at = -1.0
			_upload()
	overlay.queue_redraw()


func _input(event: InputEvent) -> void:
	if level == null or session == null:
		return
	# a finger arrives as a mouse (input_devices/pointing/emulate_mouse_from_touch)
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed:
			press(event.position)
		else:
			release()
	elif event is InputEventMouseMotion:
		drag(event.position)
	elif event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_Z:
				_press_button("undo")
			KEY_Y:
				_press_button("redo")
			KEY_R:
				_press_button("reset")
			KEY_UP:
				_press_button("prev")
			KEY_DOWN:
				_press_button("next")
			KEY_I:  # try the next set of icons
				icon_set = Icons.after(icon_set)
				note = icon_set
				note_until = now + 2.5
				overlay.queue_redraw()


func _process(delta: float) -> void:
	now += delta
	if level == null or session == null:
		return
	var dc: Dictionary = cfg["drag"]
	match phase:
		Phase.DRAG:
			if pending.x >= 0:
				if committed:
					shift = minf(1.0, shift + delta / float(dc["snap_s"]))
				else:  # follows the finger, a little behind
					shift += (drag_frac - shift) * minf(1.0, delta / float(dc["follow_s"]))
		Phase.SNAP:
			shift = minf(1.0, shift + delta / float(dc["snap_s"]))
			if shift >= 1.0:
				_apply()
		Phase.BACK:
			shift = maxf(0.0, shift - delta / float(dc["back_s"]))
			if shift <= 0.0:
				_drop()
	# eased when it runs by itself, so that the cells start and stop gently
	eased = smoothstep(0.0, 1.0, shift) if phase == Phase.SNAP and action != "play" else shift
	material_.set_shader_parameter("holes", 1.0 if eased > 0.0 and not moving.is_empty() else 0.0)
	if phase != Phase.IDLE:
		tiles.queue_redraw()
	var rw: Dictionary = cfg["reward"]
	var t := now - solved_at if solved_at >= 0.0 else -1.0
	var flat := clampf(t / float(rw["lines_s"]), 0.0, 1.0) if t >= 0.0 else 0.0
	material_.set_shader_parameter("line_alpha", 1.0 - flat)
	material_.set_shader_parameter("outline_alpha", flat)
	material_.set_shader_parameter("painting_mix", _painting_alpha())
	if t >= 0.0 and t < float(rw["lines_s"]) + float(rw["hold_s"]) + float(rw["fade_s"]) + 0.5:
		overlay.queue_redraw()
	if note_until >= 0.0 and now > note_until:
		note_until = -1.0
		overlay.queue_redraw()


func _painting_alpha() -> float:
	if solved_at < 0.0 or level.painting.is_empty():
		return 0.0
	var rw: Dictionary = cfg["reward"]
	var t := now - solved_at - float(rw["lines_s"]) - float(rw["hold_s"])
	return clampf(t / float(rw["fade_s"]), 0.0, 1.0)


# ---- the moving cells ---------------------------------------------------------------------
## Outline of the cell `from` on its way to the place and shape of the cell `to`.
func _tile(from: int, to: int, grow: float) -> PackedVector2Array:
	var a := shape_px[from]
	var b := shape_px[to]
	var c := centre_px[from].lerp(centre_px[to], eased)
	var pts := PackedVector2Array()
	pts.resize(a.size() + 1)
	pts[0] = c
	for k in a.size():
		pts[k + 1] = c + (a[k].lerp(b[k], eased) - c) * grow
	return pts


func _draw_tiles() -> void:
	if moving.is_empty() or eased <= 0.0 or level == null:
		return
	var lift := float(cfg["drag"]["lift"]) * sin(PI * eased)
	var line := _c("line")
	var width := float(cfg["picture"]["line_px"]) * unit
	# two layers: the cells of the colour that passes above are drawn last
	for layer in 2:
		var points := PackedVector2Array()
		var colours := PackedColorArray()
		var indices := PackedInt32Array()
		var rims: Array[PackedVector2Array] = []
		for k in range(0, moving.size(), 2):
			for pair in [[moving[k], moving[k + 1]], [moving[k + 1], moving[k]]]:
				var from: int = pair[0]
				var above := session.col[from] == top_colour
				if above != (layer == 1):
					continue
				var pts := _tile(from, pair[1], 1.0 + lift if above else 1.0 - lift)
				var base := points.size()
				var n := pts.size() - 1
				var col := level.colours[session.col[from]]
				points.append_array(pts)
				for j in pts.size():
					colours.append(col)
				for j in n:  # a fan around the middle
					indices.append(base)
					indices.append(base + 1 + j)
					indices.append(base + 1 + (j + 1) % n)
				var rim := pts.slice(1)
				rim.append(pts[1])
				rims.append(rim)
		if not points.is_empty():
			RenderingServer.canvas_item_add_triangle_array(tiles.get_canvas_item(), indices, points, colours)
		for rim in rims:
			tiles.draw_polyline(rim, line, width, true)


# ---- drawing ------------------------------------------------------------------------------
func _box(colour: Color) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = colour
	sb.set_corner_radius_all(roundi(float(cfg["strip"]["radius"]) * unit))
	return sb


## Lines of text centred in a box: [text, size, colour] each; the sizes shrink together if
## the lines do not fit the height.
func _lines(r: Rect2, lines: Array) -> void:
	var font := ThemeDB.fallback_font
	var inner := r.grow(-8.0 * unit)
	var align := HORIZONTAL_ALIGNMENT_CENTER
	var gap := 5.0 * unit
	var scale := 1.0
	var heights: Array[float] = []
	for attempt in 6:
		heights.clear()
		var total := -gap
		for l in lines:
			var h := font.get_multiline_string_size(l[0], align, inner.size.x, roundi(l[1] * unit * scale)).y
			heights.append(h)
			total += h + gap
		if total <= inner.size.y:
			break
		scale *= 0.88
	var sum := -gap
	for h in heights:
		sum += h + gap
	var y := inner.position.y + maxf(0.0, (inner.size.y - sum) * 0.5)
	for k in lines.size():
		var fs := roundi(lines[k][1] * unit * scale)
		overlay.draw_multiline_string(font, Vector2(inner.position.x, y + font.get_ascent(fs)),
			lines[k][0], align, inner.size.x, fs, -1, lines[k][2])
		y += heights[k] + gap


func _draw_overlay() -> void:
	if level == null or session == null:
		return
	var fg := _c("text")
	var mid := _c("mid")
	var dim := _c("dim")
	var accent := _c("accent")
	var solved := solved_at >= 0.0
	if solved:
		overlay.draw_rect(rect.grow(6.0), accent, false, 3.0)
	overlay.draw_rect(strip, _c("panel"))
	for name_ in boxes:
		overlay.draw_style_box(_box(_c("button")), boxes[name_])
	# level number (the file's name under it while levels are being chosen)
	var number: Array = [["%02d" % (at + 1), 30.0, fg], [level.id, 12.0, dim]]
	if note_until >= 0.0:
		number[1] = [note, 12.0, accent]
	_lines(boxes["number"], number)
	# moves over par
	_lines(boxes["par"], [["%d / %d" % [session.moves(), level.par], 20.0, accent if solved else fg]])
	# the painting, or the message once recovered
	var about: Array = []
	if solved:
		about.append([cfg["messages"]["solved"], 14.0, accent])
	if not level.artist.is_empty():
		about.append([level.artist, 14.0, fg])
		about.append([level.title, 13.0, mid])
		if not level.date.is_empty():
			about.append([level.date, 12.0, mid])
	_lines(boxes["info"], about)
	# buttons
	for name_ in buttons:
		var r: Rect2 = buttons[name_]
		var on := fg
		if (name_ == "undo" and session.history.is_empty()) or (name_ == "redo" and session.future.is_empty()):
			on = dim
		var hot: bool = solved and name_ == "next"
		overlay.draw_style_box(_box(accent if hot else _c("button")), r)
		var tex := Icons.texture(icon_set, name_, roundi(r.size.x * float(cfg["strip"]["icon"])))
		overlay.draw_texture(tex, (r.get_center() - tex.get_size() * 0.5).round(), _c("background") if hot else on)


# ---- headless check: solve every level by drags ---------------------------------------------
## Drags from a cell of a pair of the move to its partner; false if no pair lets itself be
## chosen that way.
func _drag_move(m: Vector3i) -> bool:
	var foam := level.foam
	for k in foam.swapping(session.col, m.x, m.y, m.z):
		for pair in [[foam.pairs_u[m.x][k], foam.pairs_v[m.x][k]], [foam.pairs_v[m.x][k], foam.pairs_u[m.x][k]]]:
			var u: int = pair[0]
			var v: int = pair[1]
			if level.cell_at((site_px[u] - rect.position) / rect.size) != u:
				continue
			var aim := (_side_mid(u, v) - site_px[u]).normalized()
			var far := site_px[u].distance_to(site_px[v]) + 3.0 * float(cfg["drag"]["min_travel_px"]) * unit
			press(site_px[u])
			drag(site_px[u] + aim * 1.5 * float(cfg["drag"]["dead_px"]) * unit)
			if drag_v == v and pending == m:
				drag(site_px[u] + aim * far)
				if committed:
					release()
					_settle()
					return true
			_drop()
	return false


func _smoke() -> void:
	var ok := true
	for k in ids.size():
		_load(k)
		var path := "res://tests/golden/%s.json" % ids[k]
		var gold: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
		var good := true
		# a short drag is only a preview: nothing is played
		var m0 := Vector3i(int(gold["moves"][0][0]), int(gold["moves"][0][1]), int(gold["moves"][0][2]))
		for kk in level.foam.swapping(session.col, m0.x, m0.y, m0.z):
			var u := level.foam.pairs_u[m0.x][kk]
			var v := level.foam.pairs_v[m0.x][kk]
			var aim := (_side_mid(u, v) - site_px[u]).normalized()
			var travel := maxf(site_px[u].distance_to(site_px[v]), float(cfg["drag"]["min_travel_px"]) * unit)
			press(site_px[u])
			drag(site_px[u] + aim * 0.45 * travel)
			release()
			_settle()
			good = good and session.moves() == 0 and session.col == level.start
			break
		for mv in gold["moves"]:
			var m := Vector3i(int(mv[0]), int(mv[1]), int(mv[2]))
			if not _drag_move(m):
				good = false
				print("SMOKE %s: no drag gives move %s" % [ids[k], m])
				break
		good = good and session.solved() and session.moves() == level.par and solved_at >= 0.0
		press(buttons["undo"].get_center())
		_settle()
		good = good and not session.solved() and session.moves() == level.par - 1
		press(buttons["redo"].get_center())
		_settle()
		good = good and session.solved()
		press(buttons["reset"].get_center())
		good = good and session.col == level.start and session.moves() == 0
		print("SMOKE %s: %s" % [ids[k], "solved at par by drags" if good else "FAILED"])
		ok = ok and good
	print("SMOKE ", "PASS" if ok else "FAIL")
	get_tree().quit(0 if ok else 1)
