__kernel void cfar_ca(__global const float *pwr,
					  __global unsigned int *mask,
					  __global float *yuzhi,
					  const float K,
					  const unsigned int guard,
					  const unsigned int train,
					  const unsigned int len)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);

	int left_edge = train + guard;
	int right_edge = len - guard - train - 1;
	if(gid < left_edge || gid > right_edge)
	{
		mask[gid] = 0;
		yuzhi[gid] = 0.0;
		return;
	}
	const int Isize = get_local_size(0);
	local float4 Ibuf[256];
	int base = get_group_id(0) * Isize;
	int idx = base + lid;

/*	for(int i = 0 ; i < (Isize + 3) / 4 ; i++)
	{
		int off = base + i * 4;
		if(off + 3 < len)
		{
			Ibuf[i] = vload4(0, pwr + off);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
*/

	float sum_left = 0.0f;
	float sum_right = 0.0f;
	for(int k = 1 ; k <= train ; ++k)
	{
		int pos_left = gid - guard - k;
		int pos_right = gid + guard + k;
		sum_left += pwr[pos_left];
		sum_right += pwr[pos_right];
	}
	float noise_level = (sum_left + sum_right) / (2.0f * train);
	float cut_pwr = pwr[gid];

	mask[gid] = (cut_pwr > K * noise_level) ? 1 : 0;
	yuzhi[gid] = K * noise_level;
}

