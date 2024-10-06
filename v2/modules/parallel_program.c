#include <pyopencl-complex.h>

float norm_squared(cfloat_t *z)
{
    return z[0].real * z[0].real + z[0].imag * z[0].imag
        + z[1].real * z[1].real + z[1].imag * z[1].imag;
}

void zs_powers(int coefficient_amount, cfloat_t *zs, cfloat_t *z)
{
    int idx = 1;
    int row = 1;
    while(idx < coefficient_amount)
    {
        int i = idx;
        int j = idx - row;
        for(; i < idx + row; i++, j++)
        {
            zs[i] = cfloat_mul(zs[j], z[0]);
        }
        zs[i] = cfloat_mul(zs[j - 1], z[1]);

        idx = i + 1;
        row++;
    }
}

void X(int coefficient_amount, __global cfloat_t *a, __global cfloat_t *b, cfloat_t *zs, cfloat_t *z, cfloat_t *x)
{
    zs_powers(coefficient_amount, zs, z);
    x[0] = cfloat_new(0,0);
    x[1] = cfloat_new(0,0);
    for(int i = 0; i < coefficient_amount; i++)
    {
        x[0] = cfloat_add(x[0], cfloat_mul(a[i], zs[i]));
        x[1] = cfloat_add(x[1], cfloat_mul(b[i], zs[i]));
    }
}

void Y(int coefficient_amount, __global cfloat_t *a, __global cfloat_t *b, cfloat_t *zs, cfloat_t *z, cfloat_t *x)
{
    X(coefficient_amount, a, b, zs, z, x);
    cfloat_t mul = cfloat_add(cfloat_mul(cfloat_conj(x[0]), z[0]), cfloat_mul(cfloat_conj(x[1]), z[1]));
    mul = cfloat_new(-mul.imag, mul.real);
    x[0] = cfloat_mul(mul, x[0]);
    x[1] = cfloat_mul(mul, x[1]);
}

void update_point(cfloat_t *z, cfloat_t *x, float epsilon)
{
    z[0] = cfloat_add(z[0], cfloat_mulr(x[0], epsilon));
    z[1] = cfloat_add(z[1], cfloat_mulr(x[1], epsilon));

    float norm = sqrt(norm_squared(z));

    z[0] = cfloat_divider(z[0], norm);
    z[1] = cfloat_divider(z[1], norm);
}

__kernel void point_trajectories(
    int coefficient_amount,
    __global cfloat_t *a,
    __global cfloat_t *b,
    int head_start,
    int frame_amount,
    int initial_conditions_size,
    float epsilon,
    int intermediate_steps,
    __global cfloat_t *initial_conditions,
    __global cfloat_t *trajectories
){
    int gid = get_global_id(0) * 2;
    int idx_step = 2;
    int idx = gid * frame_amount;

    cfloat_t z[2] = {initial_conditions[gid], initial_conditions[gid + 1]};
    cfloat_t zs[100];
    zs[0] = cfloat_new(1,0);
    cfloat_t x[2];

    for(int i = 0; i < head_start; i++)
    {
        Y(coefficient_amount, a, b, zs, z, x);
        update_point(z, x, epsilon);
    }

    for(int i = 0; i < frame_amount; i++)
    {
        trajectories[idx] = z[0];
        trajectories[idx + 1] = z[1];
        for(int j = 0; j < intermediate_steps; j++)
        {
            Y(coefficient_amount, a, b, zs, z, x);
            update_point(z, x, epsilon);
        }

        idx += idx_step;
    }
}

__kernel void find_corresponding_head(
    int coefficient_amount,
    __global cfloat_t *a,
    __global cfloat_t *b,
    int head_start,
    int initial_conditions_size,
    __global cfloat_t *orbit_heads,
    int orbit_head_amount,
    float orbit_tol,
    int max_iter,
    float epsilon,
    __global cfloat_t *initial_conditions,
    __global int *head_indices
){
    int gid = get_global_id(0);
    int idx = gid * 2;

    cfloat_t z[2] = {initial_conditions[idx], initial_conditions[idx + 1]};
    cfloat_t zs[100];
    zs[0] = cfloat_new(1,0);
    cfloat_t x[2];

    for(int i = 0; i < head_start; i++)
    {
        Y(coefficient_amount, a, b, zs, z, x);
        update_point(z, x, epsilon);
    }

    bool done = false;
    for(int i = 0; i < max_iter; i++)
    {
        Y(coefficient_amount, a, b, zs, z, x);
        update_point(z, x, epsilon);
        for(int j = 0; j < orbit_head_amount; j++)
        {
            cfloat_t diff[2] = {
                cfloat_sub(z[0], orbit_heads[j * 2]), cfloat_sub(z[1], orbit_heads[j * 2 + 1])
            };
            if(norm_squared(diff) < orbit_tol)
            {
                head_indices[gid] = j;
                return;
            }
        }
    }
    head_indices[gid] = -1;
}

__kernel void point_trajectories_not_tangent(
    int coefficient_amount,
    __global cfloat_t *a,
    __global cfloat_t *b,
    int head_start,
    int frame_amount,
    int initial_conditions_size,
    __global cfloat_t *initial_conditions,
    __global cfloat_t *trajectories,
    float epsilon,
    int intermediate_steps
){
    int gid = get_global_id(0);
    int idx = gid * 2;
    int idx_step = initial_conditions_size * 2;

    cfloat_t z[2] = {initial_conditions[idx], initial_conditions[idx + 1]};
    cfloat_t zs[100];
    zs[0] = cfloat_new(1,0);
    cfloat_t x[2];

    for(int i = 0; i < head_start; i++)
    {
        X(coefficient_amount, a, b, zs, z, x);
        update_point(z, x, epsilon);
    }

    for(int i = 0; i < frame_amount; i++)
    {
        trajectories[idx] = z[0];
        trajectories[idx + 1] = z[1];
        for(int j = 0; j < intermediate_steps; j++)
        {
            X(coefficient_amount, a, b, zs, z, x);
            update_point(z, x, epsilon);
        }

        idx += idx_step;
    }
}

__kernel void find_corresponding_head_not_tangent(
    int coefficient_amount,
    __global cfloat_t *a,
    __global cfloat_t *b,
    int head_start,
    __global cfloat_t *initial_conditions,
    __global cfloat_t *orbit_heads,
    __global int *head_indices,
    int orbit_head_amount,
    float orbit_tol,
    int max_iter,
    float epsilon
){
    int gid = get_global_id(0);
    int idx = gid * 2;

    cfloat_t z[2] = {initial_conditions[idx], initial_conditions[idx + 1]};
    cfloat_t zs[100];
    zs[0] = cfloat_new(1,0);
    cfloat_t x[2];

    for(int i = 0; i < head_start; i++)
    {
        X(coefficient_amount, a, b, zs, z, x);
        update_point(z, x, epsilon);
    }

    for(int i = 0; i < max_iter; i++)
    {
        X(coefficient_amount, a, b, zs, z, x);
        update_point(z, x, epsilon);
        for(int j = 0; j < orbit_head_amount; j++)
        {
            cfloat_t diff[2] = {
                cfloat_sub(z[0], orbit_heads[j * 2]), cfloat_sub(z[1], orbit_heads[j * 2 + 1])
            };
            if(norm_squared(diff) < orbit_tol)
            {
                head_indices[gid] = j;
                return;
            }
        }
    }
    head_indices[gid] = -1;
}