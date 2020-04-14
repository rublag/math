#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ld long double

ld ginner(ld z, int k, int m, int n, int r)
{
    ld c = powl(2, -(m+1));
    ld zcn = z*c*n;
    int tlim = (int)(fabsl(zcn)*M_E) + 1;
    int epslim = 8;
    //printf("tlim = %d\n", tlim);
    ld cur = 1./tgammal(2*r+k+1);
    ld res = cur;
    for(int t = 1; t <= tlim+epslim; ++t)
    {
        cur *= zcn/(t * (t+2*r+k));
        res += cur;
    }
    return res;
}

ld gterm(ld z, int k, int m, int n)
{
    ld c = powl(2, -(m+1));
    ld zc = z*c;
    ld eps = 1e-4;
    int rlim = (int)fabsl(zc*M_E)+1+100;
    //printf("rlim = %d\n", rlim);
    ld cur = 1;
    ld res = cur * ginner(z, k, m, n, 0);
    for(int r = 1; r < rlim || cur > eps; ++r)
    {
        cur *= (-zc*zc/2)/r;
        res += cur * ginner(z, k, m, n, r);
    }
    return res;
}

ld g0term(ld z, int k, int m, int ko)
{
    ld c = powl(2, -(m+1));
    ld zc = z*c;
    int rlim = (int)fabsl((zc*M_E)/2)+1;
    int epslim = 8;
    //printf("rlim = %d\n", rlim);
    
    //
    ld mult = tgammal(ko)/powl(2, m/2.) * (2/sqrtl(3*sqrtl(M_PI)));
    //

    ld cur = 1./tgammal(k+1) * mult;
    ld res = cur;
    printf("%Lf\t%Lf\n", cur, res);
    for(int r = 1; r <= rlim+epslim; ++r)
    {
        cur *= (-zc*zc/2)/(r* ((2*r-1)+k) * (2*r+k));
        res += cur;
        printf("%Lf\t%Lf\n", cur, res);
    }
    return res;
}

ld g(ld z, int k, int m, int n)
{
    if(n == 0)
    {
        ld sres = (g0term(z, k-1, m, k) - 1/powl(4, m+1) * g0term(z, k+1, m, k));
        printf("%Lf\n", sres);
        //ld mult = tgammal(k)/powl(2, m/2.) * (2/sqrtl(3*sqrtl(M_PI)));
        //printf("%Le\n", mult);
        //ld res = sres*mult;
        //printf("%Le\n", res);
        return sres;
    }

    ld sres = ((1-n*n)*gterm(z, k-1, m, n) + n/powl(2, m) * z * gterm(z, k, m, n) - 1./powl(4, m+1) * z*z * gterm(z, k+1, m, n));
    ld mult = tgammal(k)/powl(2, m/2.) * (2./sqrtl(3*sqrtl(M_PI))) * expl(-n*n/2.);
    ld res = sres*mult;
    return res;
}

ld gZ(ld z[], size_t len, int k, int m, int n)
{
    ld res = 0;
    for(size_t i = 0; i < len; ++i)
    {
        res += g(z[i], k, m, n);
    }
    return res/len;
}

int main(int argc, char *argv[])
{
    FILE* f = fopen(argv[1], "r");
    int len = atoi(argv[2]);

    int k = 20;
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    double c = atof(argv[5]);

    ld *z = malloc(len*sizeof(*z));
    for(int i = 0; i < len; ++i)
    {
        fscanf(f, "%Lf", &z[i]);
        z[i] /= c;
    }

    printf("%Lf\n", gZ(z, len, k, m, n));
}
