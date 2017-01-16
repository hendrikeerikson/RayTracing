#pragma OPENCL EXTENSION cl_intel_printf : enable

//
// STRUCTS FOR SCENE CREATION
//

struct Material{
    float4 color;
    float diffuse;
    float specular;
    float ambient;
    float shininess;
};


struct Sphere{
    float radius;
    float4 pos;
    uint m;
};


struct Triangle{
    float4 v0;
    float4 v1;
    float4 v2;
    float4 normal;
    uint m;
};


struct Light{
    float4 pos;
    float4 ambient;
    float4 specular;
    float4 diffuse;
};


struct Scene{
    struct Material materials[10];

    struct Sphere spheres[10];
    uint sphere_count;

    struct Triangle triangles[10];
    uint triangle_count;

    struct Light lights[1];
    uint light_count;
};

//
// UTILITY FUNCTIONS AND STRUCTS
//


struct Camera{
    float4 camera_pos;
    float4 camera_middle;
    float4 camera_up;
    float4 camera_left;
    int w_h[2];
};


struct Ray{
    float4 dir;
    float4 start;
};


uint ConvColorVec (float4 color) {
    //printf("COLOR: %f, %f, %f \n", color.x, color.y, color.z);
    color = clamp(color, 0, 1);

    uint R = (uint) (color.x * 255);
    uint G = (uint) (color.y * 255);
    uint B = (uint) (color.z * 255);

    uint RGB = (B << 16) | (G << 8) | R;

    //printf("%u \n", RGB);

    return RGB;
}


float4 reflectVec (float4 dir, float4 normal) {
    float4 refl = 2 * dot(dir, normal) * normal - dir;

    return refl;
}

//
// INTERSECTION FUNCTIONS
//

bool raySphere(struct Sphere* s, struct Ray* r, float* t)
{
	float4 rayToCenter = r->start - s->pos ;
	float dotProduct = dot(r->dir, rayToCenter);
	float d = dotProduct * dotProduct - dot(rayToCenter, rayToCenter) + s->radius * s->radius;

	if ( d < 0)
		return false;

	float result1 = -(dotProduct - sqrt(d) );
	float result2 = -(dotProduct + sqrt(d) );

    if (result1 < result2 && result1 > 0) {
        *t = result1;
        return true;
    } else if (result2 > 0) {
        *t = result2;
        return true;
    }

	return false;
}


bool rayTriangle(struct Triangle* tr, struct Ray* ray, float* t)
{
    // values for x
    float a = tr->v0.x - tr->v1.x;
    float b = tr->v0.x - tr->v2.x;
    float c = ray->dir.x;
    float d = tr->v0.x - ray->start.x;

    // values for y
    float e = tr->v0.y - tr->v1.y;
    float f = tr->v0.y - tr->v2.y;
    float g = ray->dir.y;
    float h = tr->v0.y - ray->start.y;

    // values for z
    float i = tr->v0.z - tr->v1.z;
    float j = tr->v0.z - tr->v2.z;
    float k = ray->dir.z;
    float l = tr->v0.z - ray->start.z;

    // additional values
    float m = f * k - g * j;
    float n = h * k - g * l;
    float p = f * l - h * j;
    float q = g * i - e * k;
    float s = e * j - f * i;

    float inv_denom = 1.0 / (a * m + b * q + c * s);
    float e1 = d * m - b * n - c * p;
    float beta = e1 * inv_denom;

    if (beta < 0.0){
        return false;
    }

    float r = e * l - h * i;
    float e2 = a * n + d * q + c * r;
    float gamma = e2 * inv_denom;

    if (gamma < 0.0){
        return false;
    }

    float e3 = a * p - b * r + d * s;
    float epsilon = e3 * inv_denom;

    if (gamma + beta > 1.0){
        return false;
    }

    *t = e3 * inv_denom;

    return true;
}


float Intersect (struct Ray* ray, struct Scene* scene, void** object, int* type) {
    float minT = 100000;

    for (uint i = 0; i < scene->sphere_count; i++) {
        float t;
        if (raySphere( &scene->spheres[i], ray, &t)) {
            if (t < minT) {
                minT = t;
                *type = 1;
                *object = &scene->spheres[i];
            }
        }
    }

    for (uint i = 0; i < scene->triangle_count; i++) {
        float t;
        if (rayTriangle( &scene->triangles[i], ray, &t)) {
            if (t < minT) {
                minT = t;
                *type = 2;
                *object = &scene->triangles[i];
            }
        }
    }

    return minT;
}

// currently phong
float4 BRDF (struct Material* m, float4 normal, float4 dir, float4 pos, struct Scene* scene) {
    float4 ambient;
    //float4 specular;
    float4 diffuse;

    for (uint i = 0; i < scene->light_count; i++) {
        ambient = m->ambient * scene->lights[i].ambient;

        //diffuse
        float4 l = normalize(scene->lights[i].pos - pos);
        diffuse = clamp(dot(l, normal) * m->diffuse * scene->lights[i].diffuse, 0, 1);

    }

    return (diffuse + ambient) * m->color;
}

float4 RayTrace (struct Ray* ray, struct Scene* scene) {
    void* intersect_obj = 0;
    int intersect_obj_type = 0;
    float t = Intersect(ray, scene, &intersect_obj, &intersect_obj_type);
    float4 color = {0, 0, 0, 0};

    if (t < 10000) {
        float4 point = ray->start + ray->dir * t;
        float4 normal;

        struct Material m;

        if ( intersect_obj_type == 1 ){
			normal = normalize (point - ((struct Sphere*)intersect_obj)->pos);
			m = scene->materials[((struct Sphere*)intersect_obj)->m];

			color = BRDF (&m, normal, ray->dir, point, scene);
		}

		else if ( intersect_obj_type == 2 ){
			normal = ((struct Triangle*)intersect_obj)->normal;
			m = scene->materials[((struct Triangle*)intersect_obj)->m];

			color = BRDF (&m, normal, ray->dir, point, scene);
		}
    }

    return color;

}

__kernel void colorize (__global uint *output,
                        __global struct Camera *camera_struct) {

    int gidx = get_global_id(0);
    int gidz = get_global_id(1);

    int width = camera_struct->w_h[0];
    int height = camera_struct->w_h[1];

    float4 pos = camera_struct->camera_pos;

    float4 a = camera_struct->camera_up * (gidz - height / 2);
    float4 b = camera_struct->camera_left * (width / 2 - gidx);
    float4 dir = normalize(a + b + camera_struct->camera_middle - pos);

    struct Material def_mat;
    def_mat.color = (float4){1, 0, 0, 0};
    def_mat.diffuse = 0.5f;
    def_mat.specular = 0.5f;
    def_mat.ambient = 0.5f;

    struct Scene scene;
    scene.materials[0] = def_mat;

    scene.sphere_count = 1;
    scene.spheres[0].radius = 1.0f;
    scene.spheres[0].pos = (float4){0.5, 8, -0.6, 0};
    scene.spheres[0].m = 0;

    scene.triangle_count = 1;
    scene.triangles[0].v0 = (float4){-0.5, 6, -0.5, 0};
    scene.triangles[0].v1 = (float4){0, 6, 0.5, 0};
    scene.triangles[0].v2 = (float4){0.5, 6, -0.5, 0};
    scene.triangles[0].normal = (float4){0, -1, 0, 0};
    scene.triangles[0].m = 0;

    scene.light_count = 1;
    scene.lights[0].pos = (float4){4, 4, -3, 0};
    scene.lights[0].ambient = (float4){0.2, 0.2, 0.2, 0};
    scene.lights[0].diffuse = (float4){1, 1, 1, 0};
    scene.lights[0].specular = (float4){1, 1, 1, 0};

    struct Ray ray;
    ray.dir = dir;
    ray.start = pos;

    float4 color = RayTrace(&ray, &scene);

    uint color_uint = ConvColorVec (color);
    output[gidz * width + gidx] = color_uint;
}