#include <pyopencl-complex.h>

void calcPerpVecs(cfloat_t *v, cfloat_t *iv, cfloat_t *w, cfloat_t *iw)
{
    iv[0] = cfloat_new(-v[0].imag, v[0].real);
    iv[1] = cfloat_new(-v[1].imag, v[1].real);
    w[0] = cfloat_conj(v[1]);
    w[1] = cfloat_new(-v[0].real, v[0].imag);
    iw[0] = cfloat_new(-w[0].imag, w[0].real);
    iw[1] = cfloat_new(-w[1].imag, w[1].real);
}

float scalarProduct(cfloat_t *v, cfloat_t *z)
{
    float p = v[0].real * z[0].real + v[0].imag * z[0].imag;
    p += v[1].real * z[1].real + v[1].imag * z[1].imag;

    return p;
}

void X(int N, cfloat_t *x, cfloat_t *z, __global cfloat_t *a, __global cfloat_t *b)
{
    cfloat_t s[2] = {cfloat_new(0,0), cfloat_new(0,0)};
    for(int i = 0; i < N; i++)
    {
        for(int k = 0; k < N-i; k++)
        {
            cfloat_t z0 = cfloat_powr(z[0], i);
            cfloat_t z1 = cfloat_powr(z[1], k);
            cfloat_t z0z1 = cfloat_mul(z0, z1);
            cfloat_t s_aux = cfloat_mul(a[i*N+k], z0z1);
            s[0] = cfloat_add(s[0], s_aux);
            s_aux = cfloat_mul(b[i*N+k], z0z1);
            s[1] = cfloat_add(s[1], s_aux);
        }
    }
            
    x[0] = s[0];
                
    x[1] = s[1];
}

void Z(int N, cfloat_t *x, cfloat_t *z, __global cfloat_t *a, __global cfloat_t *b)
{
    X(N, x, z, a, b);
    // float norm = sqrt(scalarProduct(x,x));
    // x[0] = cfloat_divider(x[0], norm);
    // x[1] = cfloat_divider(x[1], norm);
    // cfloat_t i = cfloat_new(0,1);
    // x[0] = cfloat_mul(z[0], i);
    // x[1] = cfloat_mul(z[1], i);
}

__kernel void updatePoints(int N, int nFrames, int sampleSize,
                __global float *p, __global cfloat_t *a, 
                __global cfloat_t *b, __global float *newp, float eps)
{
    int gid = get_global_id(0)*3;
    int idx = gid;
    newp[idx]   = p[gid];           
    newp[idx+1] = p[gid+1];
    newp[idx+2] = p[gid+2]; 
    float vnormsqrd =  p[gid]*p[gid] + p[gid + 1]*p[gid + 1] + p[gid + 2]*p[gid + 2];
    float one_minus_d = 1 - (vnormsqrd - 1)/(vnormsqrd + 1);
    cfloat_t z[2] = { cfloat_new(p[gid] * one_minus_d, p[gid + 1] * one_minus_d), 
                        cfloat_new(p[gid + 2] * one_minus_d, 1 - one_minus_d)};
    cfloat_t x[2];
    int lastidx = idx;
    for(int i = 1; i < nFrames; i++)
    {
        idx += sampleSize*3;
        for(int j = 0; j < 5; j++)
        {
            X(N, x, z, a, b);
            z[0] = cfloat_add(z[0], cfloat_mulr(x[0], eps));
            z[1] = cfloat_add(z[1], cfloat_mulr(x[1], eps));
            float normsqrd[] = {z[0].real*z[0].real + z[0].imag*z[0].imag,
                                z[1].real*z[1].real + z[1].imag*z[1].imag};
            float norm = sqrt(normsqrd[0] + normsqrd[1]);
            z[0] = cfloat_divider(z[0], norm);
            z[1] = cfloat_divider(z[1], norm);
        }
        
        newp[idx] = z[0].real/(1 - z[1].imag);
        newp[idx + 1] = z[0].imag/(1 - z[1].imag);
        newp[idx + 2] = z[1].real/(1 - z[1].imag);
    }
}

float projection(cfloat_t *v, cfloat_t *z, cfloat_t *centerPoint)
{
    cfloat_t newZ[2] = {cfloat_sub(z[0], centerPoint[0]), cfloat_sub(z[1], centerPoint[1])};
    return scalarProduct(v, newZ);
}

__kernel void calcDeadPoints(int N, int sampleSize,
                __global float *p, __global cfloat_t *a, 
                __global cfloat_t *b, __global bool *renderPoint, float eps)
{
    int gid = get_global_id(0)*3;
    int idx = get_global_id(0);
    float vnormsqrd =  p[gid]*p[gid] + p[gid + 1]*p[gid + 1] + p[gid + 2]*p[gid + 2];
    float one_minus_d = 1 - (vnormsqrd - 1)/(vnormsqrd + 1);
    cfloat_t z[2] = { cfloat_new(p[gid] * one_minus_d, p[gid + 1] * one_minus_d), 
                        cfloat_new(p[gid + 2] * one_minus_d, 1 - one_minus_d)};
    cfloat_t x[2];
    X(N, x, z, a, b);
    float zProjectionScal = scalarProduct(x, z);
    cfloat_t zProjection[2] = {cfloat_mulr(z[0], zProjectionScal), cfloat_mulr(z[1], zProjectionScal)};
    x[0] = cfloat_sub(x[0], zProjection[0]);
    x[1] = cfloat_sub(x[1], zProjection[1]);
    float norm = sqrt(scalarProduct(x, x));
    if(norm < eps) renderPoint[idx] = true;
    else renderPoint[idx] = false;
}

__kernel void updatePointsTransversal(int N, int nFrames, int sampleSize,
                __global float *p, __global cfloat_t *a, __global cfloat_t *b, 
                __global cfloat_t *centerPointHist, __global cfloat_t *vHist, __global float *newp, float eps)
{
    int idx = get_global_id(0)*3;

    cfloat_t v[2] = {vHist[0], vHist[1]};
    cfloat_t iv[2];
    cfloat_t w[2];
    cfloat_t iw[2];
    calcPerpVecs(v, iv, w, iw);

    cfloat_t centerPoint[2] = {centerPointHist[0], centerPointHist[1]};

    cfloat_t z[2];
    z[0] = cfloat_add(cfloat_add(cfloat_mulr(iv[0], p[idx]), cfloat_mulr(w[0], p[idx + 1])), cfloat_mulr(iw[0], p[idx + 2]));
    z[0] = cfloat_add(z[0], centerPoint[0]);
    z[1] = cfloat_add(cfloat_add(cfloat_mulr(iv[1], p[idx]), cfloat_mulr(w[1], p[idx + 1])), cfloat_mulr(iw[1], p[idx + 2]));
    z[1] = cfloat_add(z[1], centerPoint[1]);

    // float normsqrd[] = {z[0].real*z[0].real + z[0].imag*z[0].imag,
    //                     z[1].real*z[1].real + z[1].imag*z[1].imag};
    // float norm = sqrt(scalarProduct(z,z));

    // z[0] = cfloat_divider(z[0], norm);
    // z[1] = cfloat_divider(z[1], norm);

    // newp[idx]   = projection(iv, z, centerPoint);           
    // newp[idx+1] = projection(w, z, centerPoint);
    // newp[idx+2] = projection(iw, z, centerPoint); 
    
    cfloat_t x[2];
    cfloat_t z_centered[2];
    z_centered[0] = cfloat_sub(z[0], centerPoint[0]);
    z_centered[1] = cfloat_sub(z[1], centerPoint[1]);

    cfloat_t z_dir[2];


    float normsqrd[] = {z_centered[0].real*z_centered[0].real + z_centered[0].imag*z_centered[0].imag,
                        z_centered[1].real*z_centered[1].real + z_centered[1].imag*z_centered[1].imag};
    float norm = sqrt(scalarProduct(z_centered,z_centered));

    z_dir[0] = cfloat_divider(z_centered[0], norm);
    z_dir[1] = cfloat_divider(z_centered[1], norm);

    newp[idx]   = scalarProduct(iv, z_centered);
    newp[idx+1] = scalarProduct(w, z_centered);
    newp[idx+2] = scalarProduct(iw, z_centered);

    int vidx = 0;
    bool done = false;
    for(int i = 1; i < nFrames; i++)
    {
        vidx += 2;
        idx += sampleSize*3;
        v[0] = vHist[vidx];
        v[1] = vHist[vidx + 1];
        calcPerpVecs(v, iv, w, iw);

        centerPoint[0] = centerPointHist[vidx];
        centerPoint[1] = centerPointHist[vidx + 1];

        int it = 0;
        while(!done && scalarProduct(v, z_dir) < 0)
        {
            if(it > 10000)
            {
                done = true;
                break;
            }

            Z(N, x, z, a, b);
            z[0] = cfloat_add(z[0], cfloat_mulr(x[0], eps));
            z[1] = cfloat_add(z[1], cfloat_mulr(x[1], eps));
            // normsqrd[0] = z[0].real*z[0].real + z[0].imag*z[0].imag;
            // normsqrd[1] = z[1].real*z[1].real + z[1].imag*z[1].imag;
            // norm = sqrt(normsqrd[0] + normsqrd[1]);
            // z[0] = cfloat_divider(z[0], norm);
            // z[1] = cfloat_divider(z[1], norm);
            z_centered[0] = cfloat_sub(z[0], centerPoint[0]);
            z_centered[1] = cfloat_sub(z[1], centerPoint[1]);

            // normsqrd[0] = z_centered[0].real*z_centered[0].real + z_centered[0].imag*z_centered[0].imag;
            // normsqrd[1] = z_centered[1].real*z_centered[1].real + z_centered[1].imag*z_centered[1].imag;
            norm = sqrt(scalarProduct(z_centered,z_centered));
            z_dir[0] = cfloat_divider(z_centered[0], norm);
            z_dir[1] = cfloat_divider(z_centered[1], norm);

            it++;
        }
        
        if(done)
        {
            newp[idx]   = NAN;           
            newp[idx+1] = NAN;
            newp[idx+2] = NAN; 
        }
        else
        {
            newp[idx]   = scalarProduct(iv, z_centered);//projection(iv, z, centerPoint);           
            newp[idx+1] = scalarProduct(w, z_centered);//projection(w, z, centerPoint);
            newp[idx+2] = scalarProduct(iw, z_centered);//projection(iw, z, centerPoint); 
        }
    }
}
                
/*__kernel void updatePoints3(int N, int nFrames, int sampleSize,
                __global float *p, __global cfloat_t *a, 
                __global cfloat_t *b, float c, __global float *newp)
{
    float eps = .01;
    int gid = get_global_id(0)*3;
    int idx = gid + (nFrames-1)*sampleSize*3;
    newp[idx]   = p[gid];           
    newp[idx+1] = p[gid+1];
    newp[idx+2] = p[gid+2];          
    for(int i = nFrames-2; i >= 0; i--)
    {
        int lastidx = idx;
        cfloat_t z[2] = { cfloat_mulr(cfloat_exp(cfloat_new(0, newp[lastidx])), sin(newp[lastidx+2])), 
                            cfloat_mulr(cfloat_exp(cfloat_new(0, newp[lastidx + 1])), cos(newp[lastidx+2]))};
        cfloat_t x[2];
        Z(N, x, z, a, b);
        idx -= sampleSize*3;
        z[0] = cfloat_sub(z[0], cfloat_mulr(x[0], eps));
        z[1] = cfloat_sub(z[1], cfloat_mulr(x[1], eps));
        float normsqrd[] = {z[0].real*z[0].real + z[0].imag*z[0].imag,
                            z[1].real*z[1].real + z[1].imag*z[1].imag};
        float norm = sqrt(normsqrd[0] + normsqrd[1]);
        z[0] = cfloat_divider(z[0], norm);
        z[1] = cfloat_divider(z[1], norm);
        
        if(z[0].real != 0)
        {
            newp[idx] = atan(z[0].imag/z[0].real);
            if(z[0].real < 0)
            {
            if(z[0].imag > 0)
                newp[idx] += M_PI_F;
            else
                newp[idx] -= M_PI_F;    
            }        
        }
        else
        {
            if(z[0].imag > 0)
                newp[idx] = M_PI_2_F;
            else
                newp[idx] = -M_PI_2_F;   
        }

        if(z[1].real != 0)
        {
            newp[idx + 1] = atan(z[1].imag/z[1].real);
            if(z[1].real < 0)
            {
            if(z[1].imag > 0)
                newp[idx + 1] += M_PI_F;
            else
                newp[idx + 1] -= M_PI_F;    
            }        
        }
        else
        {
            if(z[1].imag > 0)
                newp[idx + 1] = M_PI_2_F;
            else
                newp[idx + 1] = -M_PI_2_F;   
        }
        
        newp[idx+2] = atan(sqrt(normsqrd[0]/normsqrd[1]));
    }
    
    idx = gid + (nFrames-1)*sampleSize*3;
                
    for(int i = 0; i < nFrames; i++)
    {
        int lastidx = idx;
        cfloat_t z[2] = { cfloat_mulr(cfloat_exp(cfloat_new(0, newp[lastidx])), sin(newp[lastidx+2])), 
                            cfloat_mulr(cfloat_exp(cfloat_new(0, newp[lastidx + 1])), cos(newp[lastidx+2]))};
        cfloat_t x[2];
        Z(N, x, z, a, b);
        idx += sampleSize*3;
        z[0] = cfloat_add(z[0], cfloat_mulr(x[0], eps));
        z[1] = cfloat_add(z[1], cfloat_mulr(x[1], eps));
        float normsqrd[] = {z[0].real*z[0].real + z[0].imag*z[0].imag,
                            z[1].real*z[1].real + z[1].imag*z[1].imag};
        float norm = sqrt(normsqrd[0] + normsqrd[1]);
        z[0] = cfloat_divider(z[0], norm);
        z[1] = cfloat_divider(z[1], norm);
        
        if(z[0].real != 0)
        {
            newp[idx] = atan(z[0].imag/z[0].real);
            if(z[0].real < 0)
            {
            if(z[0].imag > 0)
                newp[idx] += M_PI_F;
            else
                newp[idx] -= M_PI_F;    
            }        
        }
        else
        {
            if(z[0].imag > 0)
            newp[idx] = M_PI_2_F;
            else
            newp[idx] = -M_PI_2_F;   
        }

        if(z[1].real != 0)
        {
            newp[idx + 1] = atan(z[1].imag/z[1].real);
            if(z[1].real < 0)
            {
            if(z[1].imag > 0)
                newp[idx + 1] += M_PI_F;
            else
                newp[idx + 1] -= M_PI_F;    
            }        
        }
        else
        {
            if(z[1].imag > 0)
            newp[idx + 1] = M_PI_2_F;
            else
            newp[idx + 1] = -M_PI_2_F;   
        }
                    
        newp[idx+2] = atan(sqrt(normsqrd[0]/normsqrd[1]));
    }
}*/

__kernel void calcLineColors(int sampleSize, __global float *colorMatrix, __global float *lineColors)
{
    int gid = get_global_id(0);

    int colorMatIdx = gid % (3*sampleSize);

    lineColors[gid] = colorMatrix[colorMatIdx];
}

__kernel void calcLineConnections(int sampleSize, __global int *lineConnections, __global float *allPoints)
{
    int gid = get_global_id(0);
    int ix = gid*3;
    int iy = ix + 1;
    int iz = iy + 1;
    int idx = gid * 2;

    lineConnections[idx] = gid;
    lineConnections[idx + 1] = gid + sampleSize;
}

__kernel void calcLineConnectionsCircles(int pointsPerCircle, __global int *lineConnections, __global float *allPoints)
{
    int gid = get_global_id(0);
    int idx = gid * 2;

    lineConnections[idx] = gid;
    if(((gid + 1) % pointsPerCircle) != 0)
        lineConnections[idx + 1] = gid + 1;
    else
        lineConnections[idx + 1] = gid - pointsPerCircle + 1;
}

__kernel void calcRenderIndices(int nFrames, __global int *indices)
{
    int gid = get_global_id(0);

    if(gid < nFrames)
    {
        indices[gid] = nFrames - gid - 1;
    }
    else if(gid < 3*nFrames)
    {
        indices[gid] = gid - nFrames + 1;
    }
    else
    {
        indices[gid] = 5*nFrames - gid - 2;
    }
}