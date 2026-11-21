shader_type canvas_item;
// The foam: `labels` gives the cell of every pixel (index in R + 256 G). `cells` holds, per
// cell (column): row 0 the colour shown, row 1 the colour of the picture to recover (alpha =
// colour index), row 2 whether the cell takes part in the move under way (B = 1). The cells
// of a move are drawn by the game as moving shapes above the picture; here they are holes.

uniform sampler2D labels : filter_nearest, repeat_disable;
uniform sampler2D cells : filter_nearest, repeat_disable;
uniform sampler2D painting : filter_linear, repeat_disable;
uniform vec2 label_size = vec2(1280.0, 960.0);
uniform vec2 picture_px = vec2(920.0, 690.0);  // size of the picture on screen
uniform vec3 line_colour = vec3(0.06, 0.06, 0.08);
uniform float line_px = 1.6;  // lines between cells, screen pixels
uniform float outline_px = 3.0;  // outline of the shapes of the picture, screen pixels
uniform float line_alpha = 1.0;
uniform float outline_alpha = 0.0;
uniform float painting_mix = 0.0;
uniform vec3 hole_colour = vec3(0.07, 0.07, 0.09);
uniform float holes = 0.0;  // 1 while a move is under way

const vec2 RING[8] = vec2[](
	vec2(1.0, 0.0), vec2(0.7071, 0.7071), vec2(0.0, 1.0), vec2(-0.7071, 0.7071),
	vec2(-1.0, 0.0), vec2(-0.7071, -0.7071), vec2(0.0, -1.0), vec2(0.7071, -0.7071));

int cell_at(vec2 p) {
	ivec2 q = clamp(ivec2(floor(p)), ivec2(0), ivec2(label_size) - 1);
	vec2 t = texelFetch(labels, q, 0).rg;
	return int(t.r * 255.0 + 0.5) + 256 * int(t.g * 255.0 + 0.5);
}

// colour of `cell`; a cell that is moving leaves a hole (its shape is drawn above the picture)
vec3 shown(int cell) {
	float moving = texelFetch(cells, ivec2(cell, 2), 0).b;
	return mix(texelFetch(cells, ivec2(cell, 0), 0).rgb, hole_colour, moving * holes);
}

float wanted(int cell) {
	return texelFetch(cells, ivec2(cell, 1), 0).a;
}

void fragment() {
	vec2 p = UV * label_size;
	// colours of the four nearest pixels of the raster, blended: smooth cell borders
	vec2 q = p - 0.5;
	vec2 f = fract(q);
	vec2 b = floor(q) + 0.5;
	vec3 col = mix(
		mix(shown(cell_at(b)), shown(cell_at(b + vec2(1.0, 0.0))), f.x),
		mix(shown(cell_at(b + vec2(0.0, 1.0))), shown(cell_at(b + vec2(1.0, 1.0))), f.x), f.y);
	// share of a ring around the point that lies in another cell (or another shape)
	int here = cell_at(p);
	float want = wanted(here);
	float texel = label_size.x / picture_px.x;  // pixels of the raster per screen pixel
	float r_line = 0.5 * line_px * texel;
	float r_out = 0.5 * outline_px * texel;
	float other_cell = 0.0;
	float other_shape = 0.0;
	for (int k = 0; k < 8; k++) {
		other_cell += float(cell_at(p + r_line * RING[k]) != here);
		other_shape += float(abs(wanted(cell_at(p + r_out * RING[k])) - want) > 0.002);
	}
	col = mix(col, line_colour, line_alpha * smoothstep(0.0, 0.4, other_cell / 8.0));
	col = mix(col, line_colour, outline_alpha * smoothstep(0.0, 0.4, other_shape / 8.0));
	col = mix(col, texture(painting, UV).rgb, painting_mix);
	COLOR = vec4(col, 1.0);
}
