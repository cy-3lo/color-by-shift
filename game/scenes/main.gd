extends Control
## Play screen: the foam picture, the pointer's group and pairs, and a strip of icons.
## Options after "--": --level=ID, --smoke (headless: solve every level by clicks).
## --solved shows the level recovered (to look at the reward). In a browser:
## index.html?level=ID&solved.

const SHADER := preload("res://scenes/picture.gdshader")

var cfg: Dictionary
var ids: Array[String] = []
var at := 0
var level: FoamLevel
var session: Session

var rect := Rect2()  ## the picture on screen
var strip := Rect2()
var site_px := PackedVector2Array()
var buttons := {}  ## name -> Rect2

var picture: ColorRect
var overlay: Control
var material_: ShaderMaterial
var cell_image: Image
var cell_texture: ImageTexture

var shown := PackedColorArray()  ## colour drawn for each cell
var fade_from := PackedColorArray()
var fade_t0 := PackedFloat32Array()  ## start of each cell's cross-fade, seconds
var fading := false
var links := PackedInt32Array()  ## cells of the pairs swapped by the latest move: u, v, u, v ...
var links_t0 := -1e9
var hover_u := -1
var hover_v := -1
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
	cell_image = Image.create(n, 2, false, Image.FORMAT_RGBA8)
	shown.resize(n)
	fade_from.resize(n)
	fade_t0.resize(n)
	for i in n:
		shown[i] = level.colours[session.col[i]]
		fade_t0[i] = -1e9
		var t := level.colours[level.target[i]]
		cell_image.set_pixel(i, 1, Color(t.r, t.g, t.b, level.target[i] / 255.0))
	cell_texture = ImageTexture.create_from_image(cell_image)
	material_.set_shader_parameter("labels", ImageTexture.create_from_image(labels))
	material_.set_shader_parameter("cells", cell_texture)
	material_.set_shader_parameter("label_size", Vector2(level.label_w, level.label_h))
	var line := _c("line")
	material_.set_shader_parameter("line_colour", Vector3(line.r, line.g, line.b))
	material_.set_shader_parameter("line_px", float(cfg["picture"]["line_px"]))
	material_.set_shader_parameter("outline_px", float(cfg["picture"]["outline_px"]))
	if not level.painting.is_empty() and ResourceLoader.exists(level.painting):
		material_.set_shader_parameter("painting", load(level.painting))
	links = PackedInt32Array()
	hover_u = -1
	solved_at = -1.0
	fading = false
	if preview:
		session.col = level.target.duplicate()
		for i in n:
			shown[i] = level.colours[session.col[i]]
		solved_at = now
	_upload()
	_layout()


func _layout() -> void:
	if level == null or level.foam == null:
		return
	var size_ := get_viewport_rect().size
	var sw := float(cfg["strip"]["width"])
	var m := float(cfg["picture"]["margin"])
	var room := Vector2(size_.x - sw - 2.0 * m, size_.y - 2.0 * m)
	var w := minf(room.x, room.y * level.foam.aspect)
	var h := w / level.foam.aspect
	rect = Rect2((Vector2(size_.x - sw, size_.y) - Vector2(w, h)) * 0.5, Vector2(w, h)).abs()
	rect.position = rect.position.round()
	strip = Rect2(size_.x - sw, 0.0, sw, size_.y)
	picture.position = rect.position
	picture.size = rect.size
	material_.set_shader_parameter("px_per_texel", w / level.label_w)
	site_px.resize(level.foam.n_cells)
	for i in level.foam.n_cells:
		site_px[i] = rect.position + level.foam.sites[i] / Vector2(level.foam.aspect, 1.0) * rect.size
	var x := strip.position.x + 12.0
	var bw := sw - 24.0
	buttons = {
		"prev": Rect2(x, 14.0, 30.0, 30.0),
		"next": Rect2(x + bw - 30.0, 14.0, 30.0, 30.0),
		"undo": Rect2(x, size_.y - 118.0, bw * 0.5 - 4.0, 48.0),
		"redo": Rect2(x + bw * 0.5 + 4.0, size_.y - 118.0, bw * 0.5 - 4.0, 48.0),
		"reset": Rect2(x, size_.y - 62.0, bw, 48.0),
	}
	overlay.queue_redraw()


func _upload() -> void:
	for i in shown.size():
		cell_image.set_pixel(i, 0, shown[i])
	cell_texture.update(cell_image)


# ---- pointing and moves -------------------------------------------------------------------
## The cell under the pointer and the neighbour whose shared side is nearest to the pointer.
func pair_at(pos: Vector2) -> Vector2i:
	return level.pair_at((pos - rect.position) / rect.size * Vector2(level.foam.aspect, 1.0))


func _after(origin: Vector2) -> void:
	# cross-fade of the changed cells, starting at the click and spreading outward
	var ch := session.last_changed
	var far := 1.0
	for c in ch:
		far = maxf(far, origin.distance_to(site_px[c]))
	for c in ch:
		fade_from[c] = shown[c]
		fade_t0[c] = now + float(cfg["move"]["spread_s"]) * origin.distance_to(site_px[c]) / far
	fading = not ch.is_empty()
	links = ch
	links_t0 = now
	solved_at = now if session.solved() else -1.0
	overlay.queue_redraw()


func click(pos: Vector2) -> void:
	for name_ in buttons:
		if buttons[name_].has_point(pos):
			_press(name_)
			return
	if solved_at >= 0.0:
		return
	var p := pair_at(pos)
	if p.x >= 0 and session.play(session.move_between(p.x, p.y)):
		_after(pos)


func _press(name_: String) -> void:
	var centre := rect.get_center()
	match name_:
		"prev":
			_load(at - 1)
		"next":
			_load(at + 1)
		"undo":
			if session.undo():
				_after(centre)
		"redo":
			if session.redo():
				_after(centre)
		"reset":
			session.reset()
			for i in shown.size():
				fade_from[i] = shown[i]
				fade_t0[i] = now
			fading = true
			links = PackedInt32Array()
			solved_at = -1.0


func _input(event: InputEvent) -> void:
	if level == null or session == null:
		return
	if event is InputEventMouseMotion:
		var p := pair_at(event.position)
		if p.x != hover_u or p.y != hover_v:
			hover_u = p.x
			hover_v = p.y
			overlay.queue_redraw()
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		click(event.position)
	elif event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_Z:
				_press("undo")
			KEY_Y:
				_press("redo")
			KEY_R:
				_press("reset")
			KEY_UP:
				_press("prev")
			KEY_DOWN:
				_press("next")


func _process(delta: float) -> void:
	now += delta
	if level == null or session == null:
		return
	if fading:
		var dur := float(cfg["move"]["fade_s"])
		var busy := false
		for i in shown.size():
			var goal := level.colours[session.col[i]]
			var s := clampf((now - fade_t0[i]) / dur, 0.0, 1.0)
			if fade_t0[i] > -1e8:
				shown[i] = fade_from[i].lerp(goal, s)
				if s >= 1.0:
					fade_t0[i] = -1e9
				else:
					busy = true
		fading = busy
		_upload()
	var rw: Dictionary = cfg["reward"]
	var t := now - solved_at if solved_at >= 0.0 else -1.0
	var flat := clampf(t / float(rw["lines_s"]), 0.0, 1.0) if t >= 0.0 else 0.0
	material_.set_shader_parameter("line_alpha", 1.0 - flat)
	material_.set_shader_parameter("outline_alpha", flat)
	material_.set_shader_parameter("painting_mix", _painting_alpha())
	if t >= 0.0 or now - links_t0 < float(cfg["move"]["link_s"]) + 0.1:
		overlay.queue_redraw()


func _painting_alpha() -> float:
	if solved_at < 0.0 or level.painting.is_empty():
		return 0.0
	var rw: Dictionary = cfg["reward"]
	var t := now - solved_at - float(rw["lines_s"]) - float(rw["hold_s"])
	return clampf(t / float(rw["fade_s"]), 0.0, 1.0)


# ---- drawing ------------------------------------------------------------------------------
func _link(a: Vector2, b: Vector2, colour: Color, width: float) -> void:
	var dark := Color(0.08, 0.08, 0.1, colour.a)
	overlay.draw_line(a, b, dark, width + 2.0, true)
	overlay.draw_line(a, b, colour, width, true)


func _draw_overlay() -> void:
	if level == null or session == null:
		return
	var fg := _c("text")
	var mid := _c("mid")
	var accent := _c("accent")
	var foam := level.foam
	# the pairs just swapped
	var age := (now - links_t0) / float(cfg["move"]["link_s"])
	if age < 1.0 and solved_at < 0.0:
		var c := Color(fg, 1.0 - age)
		for k in range(0, links.size(), 2):
			_link(site_px[links[k]], site_px[links[k + 1]], c, 2.0)
	# the pointed group, the pairs that would swap, the two pointed cells
	if hover_u >= 0 and solved_at < 0.0:
		var g := foam.group_of(hover_u, hover_v)
		var m := session.move_between(hover_u, hover_v)
		for k in foam.pairs_u[g].size():
			_link(site_px[foam.pairs_u[g][k]], site_px[foam.pairs_v[g][k]], Color(mid, 0.8), 1.0)
		if m.x >= 0:
			var back := not session.history.is_empty() and session.history[-1] == m
			for k in foam.swapping(session.col, m.x, m.y, m.z):
				_link(site_px[foam.pairs_u[g][k]], site_px[foam.pairs_v[g][k]], accent if back else fg, 3.0)
		for c2 in [hover_u, hover_v]:
			overlay.draw_circle(site_px[c2], 6.0, Color(0.08, 0.08, 0.1))
			overlay.draw_circle(site_px[c2], 4.5, fg)
	if solved_at >= 0.0:
		overlay.draw_rect(rect.grow(6.0), accent, false, 3.0)
	_draw_strip(fg, mid, accent)


func _draw_strip(fg: Color, mid: Color, accent: Color) -> void:
	var font := ThemeDB.fallback_font
	var dim := _c("dim")
	overlay.draw_rect(strip, _c("panel"))
	var x := strip.position.x + 12.0
	var w := strip.size.x - 24.0
	overlay.draw_string(font, Vector2(x, 36.0), level.id, HORIZONTAL_ALIGNMENT_CENTER, w, 20, fg)
	for name_ in buttons:
		var r: Rect2 = buttons[name_]
		overlay.draw_rect(r, _c("button"))
		var c := r.get_center()
		var on := fg
		if (name_ == "undo" and session.history.is_empty()) or (name_ == "redo" and session.future.is_empty()):
			on = dim
		match name_:
			"prev":
				overlay.draw_colored_polygon([c + Vector2(-6, 0), c + Vector2(5, -7), c + Vector2(5, 7)], on)
			"next":
				overlay.draw_colored_polygon([c + Vector2(6, 0), c + Vector2(-5, -7), c + Vector2(-5, 7)], on)
			"undo", "redo":
				var s := -1.0 if name_ == "undo" else 1.0
				overlay.draw_arc(c + Vector2(0, 3), 10.0, PI, TAU, 16, on, 2.5, true)
				var tip := c + Vector2(10.0 * s, 3)
				overlay.draw_colored_polygon([tip + Vector2(-6, 0), tip + Vector2(6, 0), tip + Vector2(0, 8)], on)
			"reset":
				overlay.draw_arc(c, 10.0, -0.25 * PI, 1.5 * PI, 20, on, 2.5, true)
				var tip2 := c + Vector2(0, -10)
				overlay.draw_colored_polygon([tip2 + Vector2(-2, -6), tip2 + Vector2(-2, 6), tip2 + Vector2(7, 0)], on)
	var count := "%d / %d" % [session.moves(), level.par]
	var good := solved_at >= 0.0 and session.moves() <= level.par
	overlay.draw_string(font, Vector2(x, get_viewport_rect().size.y - 136.0), count,
		HORIZONTAL_ALIGNMENT_CENTER, w, 22, accent if good else fg)
	var a := _painting_alpha()
	if a > 0.0:
		overlay.draw_multiline_string(font, Vector2(x, 90.0), level.artist,
			HORIZONTAL_ALIGNMENT_LEFT, w, 16, -1, Color(fg, a))
		var lines := font.get_multiline_string_size(level.artist, HORIZONTAL_ALIGNMENT_LEFT, w, 16).y
		overlay.draw_multiline_string(font, Vector2(x, 100.0 + lines), level.title,
			HORIZONTAL_ALIGNMENT_LEFT, w, 14, -1, Color(mid, a))


# ---- headless check: solve every level by clicks ---------------------------------------------
func _point_for(m: Vector3i) -> Vector2:
	var foam := level.foam
	for k in foam.swapping(session.col, m.x, m.y, m.z):
		for q in level.sides[foam.edge_of(foam.pairs_u[m.x][k], foam.pairs_v[m.x][k])]:
			var pos := rect.position + q / Vector2(foam.aspect, 1.0) * rect.size
			var p := pair_at(pos)
			if p.x >= 0 and session.move_between(p.x, p.y) == m:
				return pos
	return Vector2(-1, -1)


## Share of the edges that can be pointed at from a grid of points 6 screen pixels apart.
func _pointable() -> float:
	var seen := {}
	var y := rect.position.y + 2.0
	while y < rect.end.y:
		var x := rect.position.x + 2.0
		while x < rect.end.x:
			var p := pair_at(Vector2(x, y))
			if p.x >= 0:
				seen[mini(p.x, p.y) * level.foam.n_cells + maxi(p.x, p.y)] = true
			x += 6.0
		y += 6.0
	return seen.size() / float(level.foam.edge_u.size())


func _smoke() -> void:
	var ok := true
	for k in ids.size():
		_load(k)
		var path := "res://tests/golden/%s.json" % ids[k]
		var gold: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
		var first := true
		for mv in gold["moves"]:
			var m := Vector3i(int(mv[0]), int(mv[1]), int(mv[2]))
			var pos := _point_for(m)
			if pos.x < 0.0:
				ok = false
				print("SMOKE %s: no point gives move %s" % [ids[k], m])
				break
			click(pos)
			if first and not session.solved():  # the same click again takes the move back
				click(pos)
				ok = ok and session.moves() == 0 and session.col == level.start
				click(pos)
				first = false
		var solved := session.solved() and session.moves() == level.par and solved_at >= 0.0
		click(buttons["undo"].get_center())
		solved = solved and not session.solved()
		click(buttons["redo"].get_center())
		solved = solved and session.solved()
		click(buttons["reset"].get_center())
		solved = solved and session.col == level.start
		print("SMOKE %s: %s, %.3f of the edges can be pointed at" % [ids[k], "solved at par by clicks" if solved else "FAILED", _pointable()])
		ok = ok and solved
	print("SMOKE ", "PASS" if ok else "FAIL")
	get_tree().quit(0 if ok else 1)
