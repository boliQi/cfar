__kernel void cfar_ca(__global const float *pwr,
					  __global unsigned int *mask,
					  __global float *yuzhi,
					  const float K,
					  const unsigned int guard,
					  const unsigned int train,
					  const unsigned int len)
// 新增debug数组参数
__global float *debug_ibuf,
__global float *debug_pwr,
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

	// 每个工作项负责搬运一部分数据
	int num_float4 = (1000 + 3) / 4; // pwr长度为1000
	if (lid < num_float4) {
		int off = base + lid * 4;
		if (off + 3 < 1000) {
			Ibuf[lid] = vload4(0, pwr + off);
		} else {
			// 边界处理，最后一组可能不足4个float
			float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			for (int j = 0; j < 4; ++j) {
				if (off + j < 1000) tmp[j] = pwr[off + j];
			}
			Ibuf[lid] = (float4)(tmp[0], tmp[1], tmp[2], tmp[3]);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// debug: 前16个float数据写入全局debug数组
	for (int i = 0; i < 16; ++i) {
		int ibuf_idx = (i / 4) % 256;
		int ibuf_off = i % 4;
		if (lid == 0) {
			debug_ibuf[i] = Ibuf[ibuf_idx].s[ibuf_off];
			debug_pwr[i] = pwr[i];
		}
	}


	float sum_left = 0.0f;
	float sum_right = 0.0f;
	// 使用Ibuf中的数据进行平均值计算
	for(int k = 1 ; k <= train ; ++k)
	{
		int pos_left = gid - guard - k;
		int pos_right = gid + guard + k;
		// Ibuf下标 = (pos / 4) % 256，分组内偏移 = pos % 4
		int ibuf_idx_left = (pos_left / 4) % 256;
		int ibuf_off_left = pos_left % 4;
		int ibuf_idx_right = (pos_right / 4) % 256;
		int ibuf_off_right = pos_right % 4;
		sum_left += Ibuf[ibuf_idx_left].s[ibuf_off_left];
		sum_right += Ibuf[ibuf_idx_right].s[ibuf_off_right];
	}
	int ibuf_idx_gid = (gid / 4) % 256;
	int ibuf_off_gid = gid % 4;
	float cut_pwr = Ibuf[ibuf_idx_gid].s[ibuf_off_gid];
	float noise_level = (sum_left + sum_right) / (2.0f * train);

	mask[gid] = (cut_pwr > K * noise_level) ? 1 : 0;
	yuzhi[gid] = K * noise_level;
}

