class Photon {
	Ray r;
	PVector col;

	Photon(PVector _s, 
		PVector _d, 
		PVector _col) {

		r = new Ray(_s, _d);
		col = _col;
	}
}


Photon create_photon() {
	float u = random(0.0f, 1.0f);
	float v = random(0.0f, 1.0f);

	float theta = 2 * 3.14159 * u;
	float phi = 2*v - 1;
	float aphi = acos(phi);

	Photon photon = new Photon(
		light, 
		new PVector(cos(theta) * sin(aphi), 
		sin(theta) * sin(aphi), 
		phi), 
		light_color);

	return photon;
}


void trace_photon(Photon[] pmap) {
	float t;
	Photon p;

	for (int i = 0; i < photons; i++) {
		p = pmap[i];
		t = ray_trace(p.r);

		PVector point = PVector.add(p.r.s, PVector.mult(p.r.d, t)); 
		PVector normal = new PVector(0, 0, 0);

		if (obj instanceof Sphere) {
			Sphere s = (Sphere)obj;
			p.col = s.mat.col;
			normal = s.get_normal(point);
		} else if (obj instanceof Plane) {
			Plane pl = (Plane)obj;
			p.col = pl.mat.col;
			normal = pl.get_normal(point);
		}
		
		PVector new_d = reflect(p.r.d, normal);
		p.r.d = new_d;
		p.r.s = point;
	}
}


Photon[] create_photon_map() {
	Photon[] pmap = new Photon[photons];
	Photon p;

	for (int i = 0; i < photons; i++) {
		p = create_photon();
		pmap[i] = p;
	}

	trace_photon(pmap);

	return pmap;
}
