#pragma once

#include <cmath>
#include <algorithm>

#include "../common/math_utils.h"


struct Vec3 {
    float x = 0, y = 0, z = 0;
};

extern __device__ __constant__ Vec3 aabb_min = {-1.5f, -1.5f, -1.5f};
extern __device__ __constant__ Vec3 aabb_max = {1.5f, 1.5f, 1.5f};
extern __device__ __constant__ Vec3 aabb_size = {3.0f, 3.0f, 3.0f};


/**
 * @brief Calculates the intersection of a ray with an Axis-Aligned Bounding Box (AABB).
 * * This function implements a robust method that handles rays parallel to the box axes.
 * * @param ray_o The origin of the ray.
 * @param ray_d The direction of the ray (should be normalized, but not required).
 * @param aabb_min The minimum coordinates (corner) of the AABB.
 * @param aabb_max The maximum coordinates (corner) of the AABB.
 * @param t_near Output parameter for the near intersection distance.
 * @param t_far Output parameter for the far intersection distance.
 * @return true if the ray intersects the box, false otherwise.
 */

__device__ inline bool get_ray_aabb_intersection(
    const float3& ray_o,
    const float3& ray_d,
    const float3& aabb_min,
    const float3& aabb_max,
    float& t_near,
    float& t_far
) {
    // 1. Calculate inverse direction for each component.
    const float3 inv_dir = make_float3(1.0f / ray_d.x, 1.0f / ray_d.y, 1.0f / ray_d.z);

    // 2. Calculate slab intersections for each axis individually.
    const float tx1 = (aabb_min.x - ray_o.x) * inv_dir.x;
    const float tx2 = (aabb_max.x - ray_o.x) * inv_dir.x;
    const float ty1 = (aabb_min.y - ray_o.y) * inv_dir.y;
    const float ty2 = (aabb_max.y - ray_o.y) * inv_dir.y;
    const float tz1 = (aabb_min.z - ray_o.z) * inv_dir.z;
    const float tz2 = (aabb_max.z - ray_o.z) * inv_dir.z;

    // 3. Find the latest entry point across all three slabs.
    t_near = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));

    // 4. Find the earliest exit point across all three slabs.
    t_far = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));
    
    // 5. Check for a valid intersection.
    if (t_near >= t_far || t_far < 0.0f) {
        return false;
    }

    // 6. If the ray originates inside the box, clamp the near plane to the origin.
    t_near = fmaxf(0.0f, t_near);

    return true;
}


__device__ inline float3 world_to_normalized(const float3& world_pos) {
    return make_float3(
        (world_pos.x - (-1.5f)) / 3.0f,
        (world_pos.y - (-1.5f)) / 3.0f,
        (world_pos.z - (-1.5f)) / 3.0f
    );
}

__device__ inline float3 normalized_to_world(const float3& norm_pos) {
    return make_float3(
        norm_pos.x * 3.0f - 1.5f,
        norm_pos.y * 3.0f - 1.5f,
        norm_pos.z * 3.0f - 1.5f
    );
}