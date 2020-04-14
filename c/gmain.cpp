#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <gmpxx.h>

#define ld long double
#define gf mpf_class
#define gz mpz_class


gf ginner(gf z, int k, int m, int n, int r)
{
    /* Compute inner sum for n != 0
     * 
     * Computes approximation of
     * sum_{t=1}^{\infty} (z*n / 2^(m+1))^t / (t! (t+2r+k)!) 
     * 
     *                       t
     * inf  /         -(m+1)\
     *  __  \z * n * 2      /
     * \    --------------
     * /__   t! (t+2r+k)!
     * t=0
     *
     *
     * Arguments
     * ---------
     * z : mpf_class (gmp c++ float)
     * k : int
     * m : int
     * n : int
     * r : int
     *
     * Returns
     * -------
     * mpf_class (gmp c++ float)
     *     Approximatin of the sum
     *
     * Estimation
     * ----------
     * Currently very rough.
     * Let x = z*c*n
     * Upper limit is floor(x)*e+1 + 8
     */
    mpz_class c_integer;
    mpz_ui_pow_ui(c_integer.get_mpz_t(), 2, abs(m+1)); //    cz = 2^(|m+1|)
    mpf_class c;
    c = c_integer;
    if(-(m+1) < 0)
        c = 1/c;                                //    c = 2^-(m+1)
    gf zcn = z*c*n;


    long t_limit = abs(zcn.get_si())*M_E+1 + 8; //    tlim = |zcn.to_signed_int|*e + 1

    //printf("tlim = %d\n", tlim);
    mpz_class zfactorial;                       //    int zfactorial
    mpz_fac_ui(zfactorial.get_mpz_t(), 2*r+k);  //    zfactorial = (2*r+k)!
    mpf_class factorial = zfactorial;           //    float factorial = zfactorial
    
    gf cur = 1/factorial;                       //    float cur = 1/factorial
    gf res = cur;                               //    float res = cur
    for(int t = 1; t <= t_limit; ++t)       //    res = sum_{t=1}^{tlim+epslim} zcn^t / (t! * (t+2r+k)!)
    {
        cur *= zcn/(t * (t+2*r+k));            
        res += cur;
    }                                           //    res = sum_{t=1}^{tlim+epslim} (z* 2^-(m+1) * n)^t / (t! * (t+2r+k)!)
    return res;
}


gf gterm(gf z, int k, int m, int n)
{
    /* Compute outer summation term for n != 0
     * 
     *
     */
    mpz_class cz;                                  
    mpz_ui_pow_ui(cz.get_mpz_t(), 2, abs(m+1));
    mpf_class c;
    c = cz;
    if(-(m+1) < 0)
        c = 1/c;
    gf zc = z*c;

    gf eps = 1e-4;
    int rlim = (int)fabsl(zc.get_d()*M_E)+1+100;

    //printf("rlim = %d\n", rlim);
    
    gf cur = 1;
    gf res = cur * ginner(z, k, m, n, 0);
    for(int r = 1; r < rlim || cur > eps; ++r)
    {
        cur *= (-zc*zc/2)/r;
        mpf_class tmp;
        tmp = cur*ginner(z, k, m, n, r);
        res += tmp;
    }
    return res;
}


gf g0term(gf z, int k, int m)
{
    /*
     *                           2r
     *          inf  /    -(m+1)\         r
     *           __  \ z*2      /   (-1/2)
     *          \    ----------------------
     * g0term = /__     (2r+k)!        r!
     *          r=0
     *
     */
    mpz_class cz;
    mpz_ui_pow_ui(cz.get_mpz_t(), 2, abs(m+1)); // cz = 2^|-(m+1)|
    mpf_class c;
    c = cz;
    if(-(m+1) < 0)
        c = 1/c;                                // c = 2^-(m+1)
    gf zc = z*c;                                // zc = z*c
    int rlim = (int)fabsl((z.get_d()*M_E)/2)+1;
    int epslim = 8;
    //printf("rlim = %d\n", rlim);
    mpz_class zfactorial;
    mpz_fac_ui(zfactorial.get_mpz_t(), k);      // zfactorial = k!
    mpf_class factorial = zfactorial;
    gf cur = 1 / factorial;                     // cur = 1/k!
    gf res = cur;                               // res = 1/k!
    for(int r = 1; r <= rlim+epslim; ++r)
    {
        cur *= (-zc*zc/2)/(r* ((2*r-1)+k) * (2*r+k)); // cur *= (-zc^2/2) / (r * (2r-1+k) (2r+k))
        res += cur;
    }  // res = 1/k! + sum_{r=1}^inf * (-zc^2/2)^r / (r! (2r+k)!)
    return res;
}


gf g_mn(double z, int k, int m, int n)
{
    /* Compute g_{m,n}(z)
     *
     *                                                     2    /                                                        \
     *                        1                2         -n / 2 | /   2\               n                  1              |
     * g_{m, n}(z) =  (k-1)! --------- ---------------- e       | |1-n | first_term + --- second_term - ----- third_term |
     *                               m             1/4          | \    /                m                 m+1            |
     *                        sqrt(2)   sqrt(3) * pi            \                      2                 4               /
     *
     *
     *
     *                                                  /                                \
     *                          1              2        |                 1              |
     * g_{m, 0}(z) =  (k-1)! --------- ---------------- | first_term  - ----- third_term |
     *                               m             1/4  |                 m+1            |
     *                        sqrt(2)   sqrt(3) * pi    \                4               /
     *
     */              
    
    if(n == 0)
    {

        gz second_term_power_multiplier_integer;
        mpz_ui_pow_ui(second_term_power_multiplier_integer.get_mpz_t(), 4, abs(-(m+1))); // stpmi = 4^(|-(m+1)|)
        mpf_class second_term_power_multiplier = second_term_power_multiplier_integer;
        if(-(m+1) < 0)
            second_term_power_multiplier = 1. / second_term_power_multiplier;            // stpm = 4^-(m+1) 

        gf first_term = g0term(z, k-1, m);
        gf second_term = second_term_power_multiplier * g0term(z, k+1, m);
        gf terms = first_term - second_term; 

        gz common_multiplier_power_term_integer;
        mpz_ui_pow_ui(common_multiplier_power_term_integer.get_mpz_t(), 2, abs(m));      // cmpti = 2^|m|
        gf common_multiplier_power_term = common_multiplier_power_term_integer;
        common_multiplier_power_term = sqrt(common_multiplier_power_term);               // cmpt = sqrt(2^|m|)
        if(m < 0)
            common_multiplier_power_term = 1/common_multiplier_power_term;               // cmpt = sqrt(2^m)

        gz common_multiplier_factorial_term_integer;
        mpz_fac_ui(common_multiplier_factorial_term_integer.get_mpz_t(), k-1);           // cmfti = (k-1)!
        gf common_multiplier_factorial_term = common_multiplier_factorial_term_integer; 

        gf common_multiplier = common_multiplier_factorial_term / common_multiplier_power_term * (2/sqrt(3*sqrt(M_PI)));                 
        gf res = common_multiplier * terms;
        return res;
    }

    mpz_class zpow2m, zpow4mp1, zpowmd2;
    mpf_class pow2m, pow4mp1, powmd2;

    gz second_term_power_multiplier_integer;
    mpz_ui_pow_ui(second_term_power_multiplier_integer.get_mpz_t(), 2, abs(m));  // stpmi = 2^|m|
    gf second_term_power_multiplier;
    second_term_power_multiplier = second_term_power_multiplier_integer;
    if(m < 0)
        second_term_power_multiplier = 1/second_term_power_multiplier;           // stpm = 2^-m

    gz third_term_power_multiplier_integer;
    mpz_ui_pow_ui(third_term_power_multiplier_integer.get_mpz_t(), 4, abs(m+1)); // ttpmi = 4^|m+1|
    gf third_term_power_multiplier;
    third_term_power_multiplier = third_term_power_multiplier_integer;
    if(m+1 < 0)
        third_term_power_multiplier = 1/third_term_power_multiplier;             // ttpm = 4^-(m+1)
   
    gz common_multiplier_power_part_integer;
    mpz_ui_pow_ui(common_multiplier_power_part_integer.get_mpz_t(), 2, abs(m));  // cmppi = 2^|m|
    gf common_multiplier_power_part;
    common_multiplier_power_part = common_multiplier_power_part_integer;
    common_multiplier_power_part = sqrt(common_multiplier_power_part);           // cmpp = sqrt(2^|m|)
    if(m < 0)
        common_multiplier_power_part = 1/common_multiplier_power_part;           // cmpp = sqrt(2^-m)
    
    mpz_class common_multiplier_factorial_part_integer;
    mpz_fac_ui(common_multiplier_factorial_part_integer.get_mpz_t(), k-1);       // zfactorial = (k-1)!
    mpf_class common_multiplier_factorial_part = common_multiplier_factorial_part_integer;
    
    gf common_multiplier = common_multiplier_factorial_part/common_multiplier_power_part * (2./sqrt(3*sqrt(M_PI))) * exp(-n*n/2.);
    
    gf first_term = (1-n*n)*gterm(z, k-1, m, n);
    gf second_term = n / second_term_power_multiplier * z * gterm(z, k, m, n);
    gf third_term = 1 / third_term_power_multiplier * z*z * gterm(z, k+1, m, n); 
    gf terms = first_term + second_term - third_term;

    gf res = common_multiplier * terms;
    return res;
}


double exp_val_g(double z[], size_t len, int k, int m, int n)
{
    /* Compute E(g_{m,n}(Z))
     * 
     * Parameters
     * ----------
     * z   : observations of Z
     * len : size of array z
     * k   : length of trajectories
     * m   : m-coefficient of wavelet basis
     * n   : n-coefficient of wavelet basis
     *
     * Returns
     * -------
     * Estimation of expected value
     */
    
    gf tmp = 0;
    for(size_t i = 0; i < len; ++i)
    {
        tmp += g_mn(z[i], k, m, n);
    }
    tmp /= len;
    return tmp.get_d();
}


int main(int argc, char *argv[])
{
    /* Compute estimation of E(g_{m,n}(Z))
     * 
     * Z is read from file
     * call as gmain filename number-of-zs m n
     * k is 20
     */
    FILE* f = fopen(argv[1], "r");
    int len = atoi(argv[2]);

    int k = 20;
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);

    double *z = (double*)malloc(len*sizeof(*z));
    for(int i = 0; i < len; ++i)
    {
        fscanf(f, "%lf", &z[i]);
    }

    printf("%f\n", exp_val_g(z, len, k, m, n));
}
