#include <cfar.h>

//ͳ��ʱ��,��λ��us
double time_used(struct timeval tvStart, struct timeval tvEnd)
{
    double t;
    t = (double)(1000000*(tvEnd.tv_sec-tvStart.tv_sec)+tvEnd.tv_usec-tvStart.tv_usec);
    return t;	//��λ��us
}

double time_max(double *t, int n)
{
    double ret = t[0];
    int cnt = 1;
    while(cnt < n)
    {
        if(ret < t[cnt])
        {
            ret = t[cnt];
        }
        cnt++;
    }
    return ret;
}

double time_min(double *t, int n)
{
    double ret = t[0];
    int cnt = 1;
    while(cnt < n)
    {
        if(ret > t[cnt])
        {
            ret = t[cnt];
        }
        cnt++;
    }
    return ret;
}

double time_avg(double *t, int n)
{
    double ret = t[0];
    int cnt = 1;
    while(cnt < n)
    {
        ret += t[cnt];
        cnt++;
    }
    ret /= n;
    return ret;
}
