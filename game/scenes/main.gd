extends Control
## Play screen: the foam picture and a strip of icons (a bar at the bottom when the window
## is taller than wide, as on a phone).
## The gesture: press a cell and drag toward a neighbour, with the mouse or a finger. Every
## pair of the move follows the drag; below drag.commit of the way it is a preview, beyond it
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
var info := Rect2()  ## where the count and the messages go
var unit := 1.0  ## scale of the interface (larger when the window is taller than wide)
var portrait := false
var site_px := PackedVector2Array()
var buttons := {}  ## name -> Rect2

var picture: ColorRect
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
	cell_image = Image.create(n, 4, false, Image.FORMAT_RGBA8)
	for i in n:
		var t := level.colours[level.target[i]]
		cell_image.set_pixel(i, 1, Color(t.r, t.g, t.b, level.target[i] / 255.0))
		var s := level.foam.sites[i] / Vector2(level.foam.aspect, 1.0)
		var x := clampi(roundi(s.x * 65535.0), 0, 65535)
		var y := clampi(roundi(s.y * 65535.0), 0, 65535)
		cell_image.set_pixel(i, 3, Color8(x & 255, x >> 8, y & 255, y >> 8))
	cell_texture = ImageTexture.create_from_image(cell_image)
	material_.set_shader_parameter("labels", ImageTexture.create_from_image(labels))
	material_.set_shader_parameter("cells", cell_texture)
	material_.set_shader_parameter("label_size", Vector2(level.label_w, level.label_h))
	var line := _c("line")
	material_.set_shader_parameter("line_colour", Vector3(line.r, line.g, line.b))
	material_.set_shader_parameter("front", float(cfg["drag"]["front"]))
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
		w = minf(w, (room.size.y - 2.0 * m - 150.0 * unit) * level.foam.aspect)
	var h := w / level.foam.aspect
	var top := m if portrait else (room.size.y - h) * 0.5
	rect = Rect2(Vector2((room.size.x - w) * 0.5, top).round(), Vector2(w, h))
	picture.position = rect.position
	picture.size = rect.size
	material_.set_shader_parameter("picture_px", rect.size)
	material_.set_shader_parameter("line_px", float(cfg["picture"]["line_px"]) * unit)
	material_.set_shader_parameter("outline_px", float(cfg["picture"]["outline_px"]) * unit)
	site_px.resize(level.foam.n_cells)
	for i in level.foam.n_cells:
		site_px[i] = rect.position + level.foam.sites[i] / Vector2(level.foam.aspect, 1.0) * rect.size
	var pad := 12.0 * unit
	if portrait:
		var names := ["prev", "undo", "redo", "reset", "next"]
		var bw := (strip.size.x - pad * (names.size() + 1)) / names.size()
		buttons = {}
		for k in names.size():
			buttons[names[k]] = Rect2(pad + k * (bw + pad), strip.position.y + pad, bw, strip.size.y - 2.0 * pad)
		info = Rect2(pad, rect.end.y + pad, size_.x - 2.0 * pad, strip.position.y - rect.end.y - 2.0 * pad)
	else:
		var x := strip.position.x + pad
		var bw := strip.size.x - 2.0 * pad
		buttons = {
			"prev": Rect2(x, 14.0, 34.0, 34.0),
			"next": Rect2(x + bw - 34.0, 14.0, 34.0, 34.0),
			"undo": Rect2(x, size_.y - 118.0, bw * 0.5 - 4.0, 48.0),
			"redo": Rect2(x + bw * 0.5 + 4.0, size_.y - 118.0, bw * 0.5 - 4.0, 48.0),
			"reset": Rect2(x, size_.y - 62.0, bw, 48.0),
		}
		info = Rect2(x, 64.0, bw, size_.y - 118.0 - 76.0)
	overlay.queue_redraw()


## Writes the colours of the cells (row 0) and the partners of the move under way (row 2).
func _upload() -> void:
	var n := level.foam.n_cells
	for i in n:
		cell_image.set_pixel(i, 0, level.colours[session.col[i]])
		cell_image.set_pixel(i, 2, Color(0, 0, 0, 1))
	if pending.x >= 0:
		var foam := level.foam
		for k in foam.swapping(session.col, pending.x, pending.y, pending.z):
			var u := foam.pairs_u[pending.x][k]
			var v := foam.pairs_v[pending.x][k]
			cell_image.set_pixel(u, 2, Color8(v & 255, v >> 8, 255, 255))
			cell_image.set_pixel(v, 2, Color8(u & 255, u >> 8, 255, 255))
	cell_texture.update(cell_image)


# ---- the move under way -------------------------------------------------------------------
func _set_pending(m: Vector3i, what: String) -> void:
	pending = m
	action = what
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
			_set_pending(session.move_between(drag_u, v), "play")
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
	# eased, so that the colours start and stop gently
	material_.set_shader_parameter("shift", smoothstep(0.0, 1.0, shift) if phase == Phase.SNAP else shift)
	var rw: Dictionary = cfg["reward"]
	var t := now - solved_at if solved_at >= 0.0 else -1.0
	var flat := clampf(t / float(rw["lines_s"]), 0.0, 1.0) if t >= 0.0 else 0.0
	material_.set_shader_parameter("line_alpha", 1.0 - flat)
	material_.set_shader_parameter("outline_alpha", flat)
	material_.set_shader_parameter("painting_mix", _painting_alpha())
	if t >= 0.0 and t < float(rw["lines_s"]) + float(rw["hold_s"]) + float(rw["fade_s"]) + 0.5:
		overlay.queue_redraw()


func _painting_alpha() -> float:
	if solved_at < 0.0 or level.painting.is_empty():
		return 0.0
	var rw: Dictionary = cfg["reward"]
	var t := now - solved_at - float(rw["lines_s"]) - float(rw["hold_s"])
	return clampf(t / float(rw["fade_s"]), 0.0, 1.0)


# ---- drawing ------------------------------------------------------------------------------
func _icon(name_: String, r: Rect2, colour: Color, px: float) -> void:
	var tex := Icons.texture(name_, roundi(px))
	overlay.draw_texture(tex, (r.get_center() - tex.get_size() * 0.5).round(), colour)


func _text(s: String, y: float, px: float, colour: Color) -> float:
	var font := ThemeDB.fallback_font
	var fs := roundi(px * unit)
	var align := HORIZONTAL_ALIGNMENT_CENTER
	overlay.draw_multiline_string(font, Vector2(info.position.x, y + font.get_ascent(fs)), s,
		align, info.size.x, fs, -1, colour)
	return y + font.get_multiline_string_size(s, align, info.size.x, fs).y + 6.0 * unit


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
	for name_ in buttons:
		var r: Rect2 = buttons[name_]
		var on := fg
		if (name_ == "undo" and session.history.is_empty()) or (name_ == "redo" and session.future.is_empty()):
			on = dim
		var hot: bool = solved and name_ == "next"
		overlay.draw_style_box(_box(accent if hot else _c("button"), 8.0 * unit), r)
		var icon: String = "forward" if hot else name_
		_icon(icon, r, _c("background") if hot else on, minf(r.size.x, r.size.y) * 0.62)
	# level, count, messages
	var y := info.position.y
	if not portrait:
		var font := ThemeDB.fallback_font
		overlay.draw_string(font, Vector2(info.position.x, 38.0), level.id,
			HORIZONTAL_ALIGNMENT_CENTER, info.size.x, 20, mid)
	var msg: Dictionary = cfg["messages"]
	var count := "%d / %d" % [session.moves(), level.par]
	if solved:
		y = _text(msg["solved"], y, 20.0, accent)
		var n := session.moves()
		y = _text((msg["one_move"] % level.par) if n == 1 else (msg["moves"] % [n, level.par]), y, 15.0, fg)
		var a := _painting_alpha()
		if a > 0.0:
			y = _text(level.artist, y + 10.0 * unit, 16.0, Color(fg, a))
			y = _text(level.title, y, 14.0, Color(mid, a))
	else:
		var label := count if not portrait else "%s    %s" % [level.id, count]
		_text(label, y if portrait else info.end.y - 30.0, 22.0, fg)


func _box(colour: Color, radius: float) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = colour
	sb.set_corner_radius_all(roundi(radius))
	return sb


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
			press(site_px[u])
			drag(site_px[u] + (site_px[v] - site_px[u]) * 0.3)
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
