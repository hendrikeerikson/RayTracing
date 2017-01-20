class Ray {
	PVector s;
	PVector d;

	Ray(PVector _start, PVector _dir) {
		s = _start;
		d = _dir;
	}
}


class Material {
	PVector col;
	float d;
	float s;
	float a;
	float alpha;

	Material(float r, float g, float b, 
					 float diffuse, float specular, 
					 float ambient, float shininess) {
		col = new PVector(r, g, b);
		d = diffuse;
		s = specular;
		a = ambient;
		alpha = shininess;
		
	}
}


class Sphere {
	float radius;
	PVector pos;
	Material mat;

	Sphere(float _r, PVector _pos, Material m) {
		radius = _r;
		pos = _pos;
		mat = m;
	}

	float intersect(Ray ray) {
		float t;		
		PVector rayToCenter = PVector.sub(ray.s, pos);
		float a = PVector.dot(ray.d, ray.d);
		float b = 2* PVector.dot(ray.d, rayToCenter);
		float c = PVector.dot(rayToCenter, rayToCenter) - radius * radius;
		float d = b * b - 4 * a * c;

		if (d < 0) return -1.0f;

		float result1 = (-b - sqrt(d)) / (2 * a);
		float result2 = (-b + sqrt(d)) / (2 * a);

		if (result1 < result2 && result1 > 0.001) {
			t = result1;
			return t;
		} else if (result2 > 0.001) {
			t = result2;
			return t;
		}

		return -1.0f;
	}

	PVector get_normal(PVector p) {
		return PVector.sub(p, pos).normalize();
	}
}


class Plane {
	PVector n;
	PVector p;
	Material mat;

	Plane(PVector point, PVector normal, Material m) {
		n = normal.normalize();
		p = point;		
		mat = m;
	}
	
	float intersect(Ray ray){
		float a = PVector.dot(ray.d, n);
		if (a == 0)
			return -1.0f;
			
		float b = PVector.dot(PVector.sub(p, ray.s), n);
		if (b == 0)
			return -1.0f;
			
		float c = b / a;
		if (c <= 0.001)
			return -1.0f;
			
		 return c;	
	}
	
	PVector get_normal(PVector point){
		return n;
	}
}


class Scene {
	Sphere[] spheres = new Sphere[2];
	Plane[] planes = new Plane[6];
	
	Scene() {
		Material mat1 = new Material(0.8f, 0, 0, 0.8, 0.6, 0.2, 50);
		Material mat2 = new Material(0, 0.8f, 0, 0.6, 0.3, 0.2, 10);
		Material mat3 = new Material(0, 0, 0.8f, 0.6, 0.3, 0.2, 10);
		Material mat4 = new Material(0.9f, 0.9f, 0.9f, 0.6, 0.3, 0.2, 10);
		
		spheres[0] = new Sphere(1.0f, new PVector(-1.3, 0, 5), mat1);
		spheres[1] = new Sphere(1.0f, new PVector(1.6, 0, 4), mat1);
		
		planes[0] = new Plane(new PVector(0, -1, 0), new PVector(0, 1, 0), mat4);
		planes[1] = new Plane(new PVector(0, 4, 0), new PVector(0, -1, 0), mat4);
		planes[2] = new Plane(new PVector(0, 0, 8), new PVector(0, 0, -1), mat4);
		planes[3] = new Plane(new PVector(0, 0, -2), new PVector(0, 0, 1), mat4);
		planes[4] = new Plane(new PVector(-3, 0, 0), new PVector(1, 0, 0), mat2);
		planes[5] = new Plane(new PVector(3, 0, 0), new PVector(-1, 0, 0), mat3);
	}
}
