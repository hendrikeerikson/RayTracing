Scene sc1;
float gamma = 2.2f;

// the resolution changes as the program runs
int resx, resy;
// the image is rendered a row at a time
int column;
float img_size;
Object obj;
PVector light;
PVector light_color;

int photons = 1000;
Photon[] photon_map;

boolean done_rendering = false;

///////////////////////////////////////////////////
//	SETUP THE RENDERE AND CREATE THE PHOTON MAP	 //
///////////////////////////////////////////////////

void setup() {
	size(500, 500, JAVA2D);
	frameRate(9999);

	sc1 = new Scene();

	resx = 10;	// resolution at which the current image is being rendered
	resy = 10;	// the resolution increases each time the image is completed up to a limit of 500x500
	column = 0;	// row that is currently being rendered
	img_size = 1.0f;
	light = new PVector(-2, 3.5f, 6);
	light_color = new PVector(0.95f, 0.95f, 0.95f);

	println("Starting to make the photon map");
	photon_map = create_photon_map();
	println("Photon map complete");
}

///////////////////////////////////////////////////
//	RENDER IMAGE PER LINE OR DISPLAY PHOTON MAP	 //
///////////////////////////////////////////////////

void draw() {
	int x = column * (500 / resx);
	if (!done_rendering) {
		for (int i = 0; i < resy; i++) {
			int y = i * (500 / resy);

			PVector dir = new PVector((x-250)*(img_size/500), -(y-250)*(img_size/500), 1.0);
			PVector origin = new PVector(0.0f, 0.5f, -1.5f);

			dir = dir.normalize();

			Ray ray = new Ray(origin, dir);
			float t = ray_trace(ray);
			PVector rgb = new PVector(0, 0, 0);
			PVector point = PVector.add(PVector.mult(dir, t), origin);

			rgb = colorize(point, ray);

			rgb.set(pow(rgb.x, 1/gamma), 
				pow(rgb.y, 1/gamma), 
				pow(rgb.z, 1/gamma));

			rgb.mult(255);

			stroke(rgb.x, rgb.y, rgb.z);
			fill(rgb.x, rgb.y, rgb.z);
			rect(x, y, 500/resx, 500/resy);
		}
	}
	column++;
	increase_res();

	if (done_rendering) {
		for (Photon p : photon_map) {
			int dx = 250 + (int)(500 * p.r.s.x/p.r.s.z);
			int dy = 250 + (int)(-500 * p.r.s.y/p.r.s.z);
			if (dx > 0 && dx < 500 && dy > 0 && dy < 500) {

				stroke(255*p.col.x, 255*p.col.y, 255*p.col.z);
				point(dx, dy);
			}
		}
	}
}

/////////////////////////
// ASSISTING FUNCTIONS //
/////////////////////////

PVector reflect(PVector in, PVector normal) {
	float dot = PVector.dot(in, normal);
	PVector out = PVector.sub(PVector.mult(PVector.mult(normal, dot), 2), in);
	return out;
}


float ray_trace(Ray ray) {
	float t = 999999.0f;
	boolean intersected = false;

	obj = null;

	for (Sphere sphere : sc1.spheres) {
		float dist = sphere.intersect(ray);

		if (dist > 0 && dist < t) {
			t = dist;
			obj = sphere;
			intersected = true;
		}
	}

	for (Plane plane : sc1.planes) {
		float dist = plane.intersect(ray);

		if (dist > 0 && dist < t) {
			t = dist;
			obj = plane;
			intersected = true;
		}
	}

	if (!intersected)
		return -1.0f;

	return t;
}

PVector colorize(PVector point, Ray ray) {
	PVector normal;
	PVector rgb = new PVector(0, 0, 0);

	float diff;
	float amb;
	float spec;

	if (obj instanceof Sphere) {
		Sphere s = (Sphere)obj;
		rgb = s.mat.col;
		normal = s.get_normal(point);

		if (!in_shadow(point)) {
			diff = diffuse(normal, point) * s.mat.d;
			amb = s.mat.a;
			spec = specular(normal, point, ray, s.mat.alpha) * s.mat.s;
			rgb = PVector.mult(s.mat.col, diff+amb+spec);
		} else {
			amb = s.mat.a;
			rgb = PVector.mult(s.mat.col, amb);
		}
	} else if (obj instanceof Plane) {
		Plane p = (Plane)obj;
		rgb = p.mat.col;
		normal = p.get_normal(point);

		if (!in_shadow(point)) {
			diff = diffuse(normal, point) * p.mat.d;
			amb = p.mat.a;
			spec = specular(normal, point, ray, p.mat.alpha) * p.mat.s;
			rgb = PVector.mult(p.mat.col, diff+amb+spec);
		} else {
			amb = p.mat.a;
			rgb = PVector.mult(p.mat.col, amb);
		}
	}
	return rgb;
}

float diffuse(PVector normal, PVector point) {
	PVector l = PVector.sub(light, point).normalize();
	return max(0, PVector.dot(normal, l));
}

float specular(PVector normal, PVector point, Ray ray, float alpha) {
	PVector l = PVector.sub(light, point).normalize(); 
	PVector ref_vec = reflect(l, normal);
	float dot = max(0, PVector.dot(ref_vec, PVector.mult(ray.d, -1)));
	dot = pow(dot, alpha);

	return dot;
}

boolean in_shadow(PVector point) {
	PVector dir = PVector.sub(light, point);
	float dist = dir.mag();
	dir = dir.normalize();
	PVector point2 = PVector.add(point, PVector.mult(dir, 0.003));
	Ray ray = new Ray(point, dir);

	float t = ray_trace(ray);

	if (t != -1.0f && t <= dist)
		return true;

	return false;
}

// resolution is increased 5 times everytime the image finishes rendering
void increase_res() {
	if (column == resx) {
		column = 0;
		resx *= 5;
		resy *= 5;
		if (resx > 500) {
			resx = 500;
			resy = 500;
			done_rendering = true;
		}
	}
}
