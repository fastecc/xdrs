#include"P_function.h"
#include <malloc.h>

namespace function {

	struct Multiply256LUT_t
	{
		__m256i Value[2];
	};
	static Multiply256LUT_t Multiply256LUT[256];

	GFSymbol* index_of;
	GFSymbol* alpha_to;
	GFSymbol* skewVec;//twisted factors used in FFT
	GFSymbol* B;//factors used in formal derivative
	GFSymbol* log_walsh;//factors used in the evaluation of the error locator polynomial
	GFSymbol* base;
	GFSymbol* coefL;
	GFSymbol* coefH;
	int* s;

	unsigned pack_leng;
	unsigned Trials;
	unsigned k;
	unsigned t;
	void array_params(GFSymbol* index_of1, GFSymbol* alpha_to1,
		GFSymbol* skewVec1, GFSymbol* B1,
		GFSymbol* log_walsh1, GFSymbol* base1,
		GFSymbol* coefL1, GFSymbol* coefH1,
		int* s1) {
		index_of = index_of1;
		alpha_to = alpha_to1;
		skewVec = skewVec1;
		B = B1;
		log_walsh = log_walsh1;
		base = base1;
		coefL = coefL1;
		coefH = coefH1;
		s = s1;
	}

	void main_params(unsigned pack_leng1, unsigned Trials1, unsigned k1) {
		pack_leng = pack_leng1;
		Trials = Trials1;
		k = k1;
		t = Size - k;
	}

	GFSymbol mulE(GFSymbol a, GFSymbol b) {//return a*alpha_to[b] over GF(2^r)  b!= 0
	//return a ? alpha_to[(index_of[a] + b) % mod] : 0;
		return a ? alpha_to[(index_of[a] + b & mod) + (index_of[a] + b >> len)] : 0;
	}

	void init() {//initialize index_of[], alpha_to[]
		GFSymbol mas = (1 << (len - 1)) - 1;
		GFSymbol state = 1;
		for (int i = 0; i < mod; i++) {
			alpha_to[i] = state;
			index_of[state] = i;
			if (state >> (len - 1)) {
				state &= mas;
				state = state << 1 ^ mask;
			}
			else
				state <<= 1;
		}
		alpha_to[mod] = alpha_to[0];
		index_of[0] = mod;

		//-------------------Cantor basis----------------//
		//GFSymbol mas = (1 << (len - 1)) - 1;
		//GFSymbol state = 1;
		//for (int i = 0; i < mod; i++) {
		//	alpha_to[state] = i;
		//	if (state >> (len - 1)) {
		//		state &= mas;
		//		state = state << 1 ^ mask;
		//	}
		//	else
		//		state <<= 1;
		//}
		//alpha_to[0] = mod;

		//index_of[0] = 0;
		//for (int i = 0; i < len; i++)
		//	for (int j = 0; j < 1 << i; j++)
		//		index_of[j + (1 << i)] = index_of[j] ^ CantorBase[i];
		//for (int i = 0; i < Size; i++)
		//	index_of[i] = alpha_to[index_of[i]]; //index_of[0] = mod

		//for (int i = 0; i < Size; i++)
		//	alpha_to[index_of[i]] = i;
		//alpha_to[mod] = alpha_to[0];

		//table AVX2
		// For each value we could multiply by:
		for (unsigned log_m = 0; log_m < Size; ++log_m)
		{
			// For each 4 bits of the finite field width in bits:
			for (unsigned i = 0, shift = 0; i < 2; ++i, shift += 4)
			{
				// Construct 16 entry LUT for PSHUFB
				uint8_t lut[16];
				for (uint8_t x = 0; x < 16; ++x)
					//将一个8bit的数分为高4位和低4位
					//每4bit都有16种情况
					//a*c=(al+ah*x^4)*c=al*c+(ah*x^4)*c
				{
					lut[x] = mulE(x << shift, static_cast<uint8_t>(log_m));
				}
				const __m128i* v_ptr = reinterpret_cast<const __m128i*>(&lut[0]);
				const __m128i value = _mm_loadu_si128(v_ptr);

				// Store in 256-bit wide table
				_mm256_storeu_si256((__m256i*) & Multiply256LUT[log_m].Value[i], _mm256_broadcastsi128_si256(value));
			}
		}
	}

	void init_dec() {//initialize skewVec[], B[], index_of_walsh[]
		for (int i = 1; i < len; i++)
			base[i - 1] = 1 << i;

		for (int m = 0; m < len - 1; m++) {
			int step = 1 << (m + 1);
			skewVec[(1 << m) - 1] = 0;
			for (int i = m; i < len - 1; i++) {
				int s = 1 << (i + 1);
				for (int j = (1 << m) - 1; j < s; j += step)
				{
					skewVec[j + s] = skewVec[j] ^ base[i];
				}
			}

			base[m] = mod - index_of[mulE(base[m], index_of[base[m] ^ 1])];
			for (int i = m + 1; i < len - 1; i++)
				base[i] = mulE(base[i], (index_of[base[i] ^ 1] + base[m]) % mod);
		}
		for (int i = 0; i < Size; i++)
			skewVec[i] = index_of[skewVec[i]];

		for (int i = 0; i < len; i++)
		{
			s[i] = mod - i;
			for (int j = 1; j < (1 << i); j++)
			{
				s[i] += (mod - index_of[(1 << i) ^ j] + index_of[j]);
			}
			//s[i] = (mod + s[i]) % mod;
		}

		B[0] = 0;
		for (int i = 0; i < len; i++) {
			int depart = 1 << i;
			for (int j = 0; j < depart; j++)
				B[j + depart] = (B[j] + s[i]) % mod;
		}
		//for (int i = 0; i < 255; i++)printf("%d", B[i]);
		memcpy(log_walsh, index_of, Size * sizeof(GFSymbol));
		log_walsh[0] = 0;
		walsh(log_walsh, Size);


		int a = 0;
		for (int i = 1; i < k; i++)
		{
			a += index_of[i];
		}
		for (int j = 1; j < Size / k; j++)
		{
			int b = 0;
			for (int i = 0; i < k; i++)
			{
				b += index_of[(j * k) ^ i];
			}
			coefL[j - 1] = (mod + a - (b % mod)) % mod;
		}

		int c = 1;
		int t = Size - k;
		for (int j = t; j < Size; j = j << 1)
			for (int i = 0; i < j; i++)
				c = mulE(c, index_of[j ^ i]);

		for (int i = t; i < Size; i++)
			coefH[i] = c;

		for (int j = t; j < Size; j++)
			for (int i = 0; i < t; i++)
				coefH[j] = mulE(coefH[j], index_of[j ^ i]);

		for (int i = t; i < Size; i++)
			coefH[i] = mod - index_of[coefH[i]];


	}

	void walsh(GFSymbol* data, int size) {//fast Walsh–Hadamard transform over modulo mod
		for (int depart_no = 1; depart_no < size; depart_no <<= 1) {
			for (int j = 0; j < size; j += depart_no << 1) {
				for (int i = j; i < depart_no + j; i++) {
					unsigned tmp2 = data[i] + mod - data[i + depart_no];
					data[i] = (data[i] + data[i + depart_no] & mod) + ((data[i] + data[i + depart_no]) >> len);
					data[i + depart_no] = (tmp2 & mod) + (tmp2 >> len);
				}
			}
		}
		return;
	}

	static void mul_mem(GFSymbol* x, const GFSymbol* y, GFSymbol log_m, uint64_t bytes) {//x<-y*log_m,bytes:pack_leng
		const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[log_m].Value[0]);
		const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[log_m].Value[1]);
		const __m256i clr_mask = _mm256_set1_epi8(0x0f);

		__m256i* x32 = reinterpret_cast<__m256i*>(x);
		const __m256i* y32 = reinterpret_cast<const __m256i*>(y);
		while (bytes > 0) {
			__m256i data = _mm256_loadu_si256(y32);
			__m256i lo = _mm256_and_si256(data, clr_mask);
			lo = _mm256_shuffle_epi8(table_lo_y, lo);
			__m256i hi = _mm256_srli_epi64(data, 4);
			hi = _mm256_and_si256(hi, clr_mask);
			hi = _mm256_shuffle_epi8(table_hi_y, hi);
			_mm256_storeu_si256(x32, _mm256_xor_si256(lo, hi));
			x32 += 1;
			y32 += 1;
			bytes -= 32;
		}
	}

	//------------------------------------------------------------------------------
	// XOR Memory

	void xor_mem(GFSymbol* x, const GFSymbol* y, uint64_t bytes)
	{
		__m256i* x32 = reinterpret_cast<__m256i*>(x);
		const __m256i* y32 = reinterpret_cast<const __m256i*>(y);
		while (bytes >= 128)
		{

			const __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
			const __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
			const __m256i x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2), _mm256_loadu_si256(y32 + 2));
			const __m256i x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3), _mm256_loadu_si256(y32 + 3));
			_mm256_storeu_si256(x32, x0);
			_mm256_storeu_si256(x32 + 1, x1);
			_mm256_storeu_si256(x32 + 2, x2);
			_mm256_storeu_si256(x32 + 3, x3);
			x32 += 4, y32 += 4;
			bytes -= 128;
		};
		if (bytes > 0)
		{
			const __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
			const __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
			_mm256_storeu_si256(x32, x0);
			_mm256_storeu_si256(x32 + 1, x1);
		}
		return;
	}
	// LEO_TRY_AVX2

	void xor_mem4(
		void* vx_0, const void* vy_0,
		void* vx_1, const void* vy_1,
		void* vx_2, const void* vy_2,
		void* vx_3, const void* vy_3,
		uint64_t bytes)
	{
		__m256i* x32_0 = reinterpret_cast<__m256i*>      (vx_0);
		const __m256i* y32_0 = reinterpret_cast<const __m256i*>(vy_0);
		__m256i* x32_1 = reinterpret_cast<__m256i*>      (vx_1);
		const __m256i* y32_1 = reinterpret_cast<const __m256i*>(vy_1);
		__m256i* x32_2 = reinterpret_cast<__m256i*>      (vx_2);
		const __m256i* y32_2 = reinterpret_cast<const __m256i*>(vy_2);
		__m256i* x32_3 = reinterpret_cast<__m256i*>      (vx_3);
		const __m256i* y32_3 = reinterpret_cast<const __m256i*>(vy_3);
		while (bytes >= 128)
		{
			const __m256i x0_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0), _mm256_loadu_si256(y32_0));
			const __m256i x1_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 1), _mm256_loadu_si256(y32_0 + 1));
			const __m256i x2_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 2), _mm256_loadu_si256(y32_0 + 2));
			const __m256i x3_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 3), _mm256_loadu_si256(y32_0 + 3));
			_mm256_storeu_si256(x32_0, x0_0);
			_mm256_storeu_si256(x32_0 + 1, x1_0);
			_mm256_storeu_si256(x32_0 + 2, x2_0);
			_mm256_storeu_si256(x32_0 + 3, x3_0);
			x32_0 += 4, y32_0 += 4;
			const __m256i x0_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1), _mm256_loadu_si256(y32_1));
			const __m256i x1_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 1), _mm256_loadu_si256(y32_1 + 1));
			const __m256i x2_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 2), _mm256_loadu_si256(y32_1 + 2));
			const __m256i x3_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 3), _mm256_loadu_si256(y32_1 + 3));
			_mm256_storeu_si256(x32_1, x0_1);
			_mm256_storeu_si256(x32_1 + 1, x1_1);
			_mm256_storeu_si256(x32_1 + 2, x2_1);
			_mm256_storeu_si256(x32_1 + 3, x3_1);
			x32_1 += 4, y32_1 += 4;
			const __m256i x0_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2), _mm256_loadu_si256(y32_2));
			const __m256i x1_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 1), _mm256_loadu_si256(y32_2 + 1));
			const __m256i x2_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 2), _mm256_loadu_si256(y32_2 + 2));
			const __m256i x3_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 3), _mm256_loadu_si256(y32_2 + 3));
			_mm256_storeu_si256(x32_2, x0_2);
			_mm256_storeu_si256(x32_2 + 1, x1_2);
			_mm256_storeu_si256(x32_2 + 2, x2_2);
			_mm256_storeu_si256(x32_2 + 3, x3_2);
			x32_2 += 4, y32_2 += 4;
			const __m256i x0_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3), _mm256_loadu_si256(y32_3));
			const __m256i x1_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 1), _mm256_loadu_si256(y32_3 + 1));
			const __m256i x2_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 2), _mm256_loadu_si256(y32_3 + 2));
			const __m256i x3_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 3), _mm256_loadu_si256(y32_3 + 3));
			_mm256_storeu_si256(x32_3, x0_3);
			_mm256_storeu_si256(x32_3 + 1, x1_3);
			_mm256_storeu_si256(x32_3 + 2, x2_3);
			_mm256_storeu_si256(x32_3 + 3, x3_3);
			x32_3 += 4, y32_3 += 4;
			bytes -= 128;
		}
		if (bytes > 0)
		{
			const __m256i x0_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0), _mm256_loadu_si256(y32_0));
			const __m256i x1_0 = _mm256_xor_si256(_mm256_loadu_si256(x32_0 + 1), _mm256_loadu_si256(y32_0 + 1));
			const __m256i x0_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1), _mm256_loadu_si256(y32_1));
			const __m256i x1_1 = _mm256_xor_si256(_mm256_loadu_si256(x32_1 + 1), _mm256_loadu_si256(y32_1 + 1));
			_mm256_storeu_si256(x32_0, x0_0);
			_mm256_storeu_si256(x32_0 + 1, x1_0);
			_mm256_storeu_si256(x32_1, x0_1);
			_mm256_storeu_si256(x32_1 + 1, x1_1);
			const __m256i x0_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2), _mm256_loadu_si256(y32_2));
			const __m256i x1_2 = _mm256_xor_si256(_mm256_loadu_si256(x32_2 + 1), _mm256_loadu_si256(y32_2 + 1));
			const __m256i x0_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3), _mm256_loadu_si256(y32_3));
			const __m256i x1_3 = _mm256_xor_si256(_mm256_loadu_si256(x32_3 + 1), _mm256_loadu_si256(y32_3 + 1));
			_mm256_storeu_si256(x32_2, x0_2);
			_mm256_storeu_si256(x32_2 + 1, x1_2);
			_mm256_storeu_si256(x32_3, x0_3);
			_mm256_storeu_si256(x32_3 + 1, x1_3);
		}
		return;
	}

	void VectorXOR(
		unsigned count,
		GFSymbol** x,
		GFSymbol** y)
	{
		if (count >= 4)
		{
			int i_end = count - 4;
			for (int i = 0; i <= i_end; i += 4)
			{
				xor_mem4(
					x[i + 0], y[i + 0],
					x[i + 1], y[i + 1],
					x[i + 2], y[i + 2],
					x[i + 3], y[i + 3],
					 pack_leng);
			}
			count %= 4;
			i_end -= count;
			x += i_end;
			y += i_end;
		}
		for (unsigned i = 0; i < count; ++i)
			xor_mem(x[i], y[i], pack_leng);
	}

	void formal_derivative(GFSymbol** cos, int size) {//formal derivative of polynomial in the new basis

		for (int i = 0; i < size; i++) {//cantor基可省略
			mul_mem(cos[i], cos[i], B[i], sizeof(GFSymbol) *pack_leng);
		}
		for (int i = 1; i < size; i++) {

			memset(cos[i - 1], 0, sizeof(GFSymbol) * pack_leng);
			int leng = ((i ^ i - 1) + 1) >> 1;
			for (int j = i - leng; j < i; j++) {
				xor_mem(cos[j], cos[j + leng], pack_leng);
			}
		}
		memset(cos[size - 1], 0, sizeof(GFSymbol) * pack_leng);

		for (int i = 0; i < size; i++) {
			mul_mem(cos[i], cos[i], mod - B[i],pack_leng);
		}
		return;
	}

	void LCH(GFSymbol** data, int size, int index) {//FFT in the proposed basis     size=data的行数
		const __m256i clr_mask = _mm256_set1_epi8(0x0f);

		for (int depart_no = size >> 1; depart_no > 0; depart_no >>= 1) {
			for (int j = depart_no; j < size; j += depart_no << 1) {
				GFSymbol skew = skewVec[j + index - 1];

				const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
				const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

				for (int i = j - depart_no; i < j; i++) {
					__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);

						if (skew != mod) {
							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
						}
						_mm256_storeu_si256(x32, x_data);

						y_data = _mm256_xor_si256(y_data, x_data);
						_mm256_storeu_si256(y32, y_data);

						x32 += 1;
						y32 += 1;
					}
				}
			}
		}
		return;
	}

	void LCH4(GFSymbol** data, int size, int index) {//FFT in the proposed basis     size=data的行数
		const __m256i clr_mask = _mm256_set1_epi8(0x0f);
		int depart_no_4;
		for (int depart_no = size >> 2; depart_no > 0; depart_no_4 = depart_no, depart_no >>= 2) {

			for (int j = depart_no; j < size; j += depart_no << 2) {

				GFSymbol skew01 = skewVec[j + index - 1];
				GFSymbol skew02 = skewVec[j + depart_no + index - 1];
				GFSymbol skew23 = skewVec[j + depart_no * 2 + index - 1];

				for (int i = j - depart_no; i < j; i++) {
					const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
					const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
					const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
					const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
					const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
					const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);
					__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);
					__m256i* z32 = reinterpret_cast<__m256i*>(data[i + depart_no * 2]);
					__m256i* w32 = reinterpret_cast<__m256i*>(data[i + depart_no * 3]);
					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);
						__m256i z_data = _mm256_loadu_si256(z32);
						__m256i w_data = _mm256_loadu_si256(w32);
						if (skew02 != mod) {
							__m256i lo = _mm256_and_si256(z_data, clr_mask);
							lo = _mm256_shuffle_epi8(table02_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(z_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table02_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

							lo = _mm256_and_si256(w_data, clr_mask);
							lo = _mm256_shuffle_epi8(table02_lo_y, lo);
							hi = _mm256_srli_epi64(w_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table02_hi_y, hi);
							y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));

						}
						z_data = _mm256_xor_si256(x_data, z_data);
						w_data = _mm256_xor_si256(y_data, w_data);

						if (skew01 != mod) {
							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table01_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table01_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
						}
						y_data = _mm256_xor_si256(x_data, y_data);

						_mm256_storeu_si256(x32, x_data);
						_mm256_storeu_si256(y32, y_data);
						x32 += 1;
						y32 += 1;

						if (skew23 != mod) {
							__m256i lo = _mm256_and_si256(w_data, clr_mask);
							lo = _mm256_shuffle_epi8(table23_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(w_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table23_hi_y, hi);
							z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
						}
						w_data = _mm256_xor_si256(z_data, w_data);

						_mm256_storeu_si256(z32, z_data);
						_mm256_storeu_si256(w32, w_data);
						z32 += 1;
						w32 += 1;

					}
				}
			}
		}
		if (depart_no_4 == 2) {
			for (int j = 0; j < size; j += 2) {
				GFSymbol skew = skewVec[j + index];
				if (skew == mod) {
					xor_mem(data[j + 1], data[j],  pack_leng);
				}
				else {
					const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
					const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

					__m256i* x32 = reinterpret_cast<__m256i*>(data[j]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[j + 1]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);

						__m256i lo = _mm256_and_si256(y_data, clr_mask);
						lo = _mm256_shuffle_epi8(table_lo_y, lo);
						__m256i hi = _mm256_srli_epi64(y_data, 4);
						hi = _mm256_and_si256(hi, clr_mask);
						hi = _mm256_shuffle_epi8(table_hi_y, hi);
						x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

						y_data = _mm256_xor_si256(y_data, x_data);
						_mm256_storeu_si256(x32, x_data);
						_mm256_storeu_si256(y32, y_data);

						x32 += 1;
						y32 += 1;
					}

				}

			}
		}

		return;
	}

	void ILCH(GFSymbol** data, int size, int index) {//IFFT in the proposed basis

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);

		for (int depart_no = 1; depart_no < size; depart_no <<= 1) {
			for (int j = depart_no; j < size; j += depart_no << 1) {
				GFSymbol skew = skewVec[j + index - 1];

				const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
				const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

				for (int i = j - depart_no; i < j; i++) {
					__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);

						y_data = _mm256_xor_si256(y_data, x_data);
						_mm256_storeu_si256(y32, y_data);

						if (skew != mod) {
							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
						}
						_mm256_storeu_si256(x32, x_data);

						x32 += 1;
						y32 += 1;
					}
				}
			}
		}
		return;
	}
	void ILCH4(GFSymbol** data, int size, int index) {//IFFT in the proposed basis

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);
		int depart_no_4, depart_no;
		for (depart_no = 1, depart_no_4 = 4; depart_no_4 <= size; depart_no = depart_no_4, depart_no_4 <<= 2) {

			for (int j = depart_no; j <= size; j += depart_no << 2) {

				GFSymbol skew01 = skewVec[j + index - 1];
				GFSymbol skew02 = skewVec[j + depart_no + index - 1];
				GFSymbol skew23 = skewVec[j + depart_no * 2 + index - 1];
				for (int i = j - depart_no; i < j; i++) {

					const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
					const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
					const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
					const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
					const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
					const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);

					__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);
					__m256i* z32 = reinterpret_cast<__m256i*>(data[i + depart_no * 2]);
					__m256i* w32 = reinterpret_cast<__m256i*>(data[i + depart_no * 3]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);

						y_data = _mm256_xor_si256(x_data, y_data);
						if (skew01 != mod) {
							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table01_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table01_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
						}
						__m256i z_data = _mm256_loadu_si256(z32);
						__m256i w_data = _mm256_loadu_si256(w32);

						w_data = _mm256_xor_si256(z_data, w_data);

						if (skew23 != mod) {
							__m256i lo = _mm256_and_si256(w_data, clr_mask);
							lo = _mm256_shuffle_epi8(table23_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(w_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table23_hi_y, hi);
							z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
						}

						z_data = _mm256_xor_si256(x_data, z_data);
						w_data = _mm256_xor_si256(y_data, w_data);

						if (skew02 != mod) {
							__m256i lo = _mm256_and_si256(z_data, clr_mask);
							lo = _mm256_shuffle_epi8(table02_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(z_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table02_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

							lo = _mm256_and_si256(w_data, clr_mask);
							lo = _mm256_shuffle_epi8(table02_lo_y, lo);
							hi = _mm256_srli_epi64(w_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table02_hi_y, hi);
							y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));
						}

						_mm256_storeu_si256(x32, x_data);
						_mm256_storeu_si256(y32, y_data);
						_mm256_storeu_si256(z32, z_data);
						_mm256_storeu_si256(w32, w_data);

						x32 += 1;
						y32 += 1;
						z32 += 1;
						w32 += 1;
					}
				}
			}
		}
		if (depart_no < size) {
			const GFSymbol skew = skewVec[depart_no + index - 1];
			if (skew == mod) {
				VectorXOR(depart_no, data + depart_no, data);
			}
			else {
				for (int j = 0; j < depart_no; j++) {

					const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
					const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);


					__m256i* x32 = reinterpret_cast<__m256i*>(data[j]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[j + depart_no]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);
						y_data = _mm256_xor_si256(y_data, x_data);
						_mm256_storeu_si256(y32, y_data);

						__m256i lo = _mm256_and_si256(y_data, clr_mask);
						lo = _mm256_shuffle_epi8(table_lo_y, lo);
						__m256i hi = _mm256_srli_epi64(y_data, 4);
						hi = _mm256_and_si256(hi, clr_mask);
						hi = _mm256_shuffle_epi8(table_hi_y, hi);
						x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

						_mm256_storeu_si256(x32, x_data);

						x32 += 1;
						y32 += 1;
					}
				}
			}
		}
		return;
	}
	void ILCH_cpy(const GFSymbol** data, GFSymbol**parity, int size, int index) {//IFFT in the proposed basis

		for (unsigned i = 0; i < size; ++i)
			memcpy(parity[i], data[i], sizeof(GFSymbol) * pack_leng);

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);

		for (int depart_no = 1; depart_no < size; depart_no <<= 1) {
			for (int j = depart_no; j < size; j += depart_no << 1) {
				GFSymbol skew = skewVec[j + index - 1];

				const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
				const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

				for (int i = j - depart_no; i < j; i++) {
					__m256i* x32 = reinterpret_cast<__m256i*>(parity[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(parity[i + depart_no]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);

						y_data = _mm256_xor_si256(y_data, x_data);
						_mm256_storeu_si256(y32, y_data);

						if (skew != mod) {
							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
						}
						_mm256_storeu_si256(x32, x_data);

						x32 += 1;
						y32 += 1;
					}
				}
			}
		}
		return;
	}

	void ILCH4_cpy(const GFSymbol** data, GFSymbol**parity, int size, int index) {//IFFT in the proposed basis

		for (unsigned i = 0; i < size; ++i)
			memcpy(parity[i], data[i], sizeof(GFSymbol) * pack_leng);

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);
		int depart_no_4, depart_no;
		for (depart_no = 1, depart_no_4 = 4; depart_no_4 <= size; depart_no = depart_no_4, depart_no_4 <<= 2) {

			for (int j = depart_no; j <= size; j += depart_no << 2) {
				GFSymbol skew01 = skewVec[j + index - 1];
				GFSymbol skew02 = skewVec[j + depart_no + index - 1];
				GFSymbol skew23 = skewVec[j + depart_no * 2 + index - 1];

				for (int i = j - depart_no; i < j; i++) {

					const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
					const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
					const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
					const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
					const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
					const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);

					__m256i* x32 = reinterpret_cast<__m256i*>(parity[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(parity[i + depart_no]);
					__m256i* z32 = reinterpret_cast<__m256i*>(parity[i + depart_no * 2]);
					__m256i* w32 = reinterpret_cast<__m256i*>(parity[i + depart_no * 3]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);

						y_data = _mm256_xor_si256(x_data, y_data);
						if (skew01 != mod) {
							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table01_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table01_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
						}
						__m256i z_data = _mm256_loadu_si256(z32);
						__m256i w_data = _mm256_loadu_si256(w32);

						w_data = _mm256_xor_si256(z_data, w_data);
						
						if (skew23 != mod) {
							__m256i lo = _mm256_and_si256(w_data, clr_mask);
							lo = _mm256_shuffle_epi8(table23_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(w_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table23_hi_y, hi);
							z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
						}

						z_data = _mm256_xor_si256(x_data, z_data);
						w_data = _mm256_xor_si256(y_data, w_data);

						if (skew02 != mod) {
							__m256i lo = _mm256_and_si256(z_data, clr_mask);
							lo = _mm256_shuffle_epi8(table02_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(z_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table02_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));


							lo = _mm256_and_si256(w_data, clr_mask);
							lo = _mm256_shuffle_epi8(table02_lo_y, lo);
							hi = _mm256_srli_epi64(w_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table02_hi_y, hi);
							y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));
						}

						_mm256_storeu_si256(x32, x_data);
						_mm256_storeu_si256(y32, y_data);
						_mm256_storeu_si256(z32, z_data);
						_mm256_storeu_si256(w32, w_data);

						x32 += 1;
						y32 += 1;
						z32 += 1;
						w32 += 1;
					}
				}
			}
		}
		if (depart_no < size) {
			GFSymbol skew = skewVec[depart_no + index - 1];
			if (skew == mod) {
				VectorXOR(depart_no, parity + depart_no, parity);
			}
			else {
				for (int j = 0; j < depart_no; j++) {

					const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
					const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

					__m256i* x32 = reinterpret_cast<__m256i*>(parity[j]);
					__m256i* y32 = reinterpret_cast<__m256i*>(parity[j + depart_no]);

					for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
						__m256i x_data = _mm256_loadu_si256(x32);
						__m256i y_data = _mm256_loadu_si256(y32);
						y_data = _mm256_xor_si256(y_data, x_data);
						_mm256_storeu_si256(y32, y_data);

						__m256i lo = _mm256_and_si256(y_data, clr_mask);
						lo = _mm256_shuffle_epi8(table_lo_y, lo);
						__m256i hi = _mm256_srli_epi64(y_data, 4);
						hi = _mm256_and_si256(hi, clr_mask);
						hi = _mm256_shuffle_epi8(table_hi_y, hi);
						x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

						_mm256_storeu_si256(x32, x_data);

						x32 += 1;
						y32 += 1;

					}
				}
			}
		}
		return;
	}

	void ILCH_cpy_xor(const GFSymbol** data, GFSymbol**parity, GFSymbol**xor_result, int size, int index) {//IFFT in the proposed basis
		for (unsigned i = 0; i < size; ++i)
			memcpy(parity[i], data[i], sizeof(GFSymbol) * pack_leng);

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);

		for (int depart_no = 1; depart_no < size; depart_no <<= 1) {
			for (int j = depart_no; j < size; j += depart_no << 1) {
				GFSymbol skew = skewVec[j + index - 1];

				const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
				const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

				for (int i = j - depart_no; i < j; i++) {
					__m256i* x32 = reinterpret_cast<__m256i*>(parity[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(parity[i + depart_no]);
					__m256i* x32_xor_result = reinterpret_cast<__m256i*>(xor_result[i]);
					__m256i* y32_xor_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no]);
					if (depart_no == size / 2)
					{
						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);
							__m256i x_data_xor_result = _mm256_loadu_si256(x32_xor_result);
							__m256i y_data_xor_result = _mm256_loadu_si256(y32_xor_result);

							y_data = _mm256_xor_si256(y_data, x_data);
							_mm256_storeu_si256(y32, y_data);

							if (skew != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							_mm256_storeu_si256(x32, x_data);

							x_data_xor_result = _mm256_xor_si256(x_data, x_data_xor_result);
							y_data_xor_result = _mm256_xor_si256(y_data, y_data_xor_result);

							_mm256_storeu_si256(x32_xor_result, x_data_xor_result);
							_mm256_storeu_si256(y32_xor_result, y_data_xor_result);

							x32 += 1;
							y32 += 1;
							x32_xor_result += 1;
							y32_xor_result += 1;
						}
					}
					else {
						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);

							y_data = _mm256_xor_si256(y_data, x_data);
							_mm256_storeu_si256(y32, y_data);

							if (skew != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							_mm256_storeu_si256(x32, x_data);

							x32 += 1;
							y32 += 1;
						}
					}
				}
			}
		}
		return;
	}

	void ILCH4_cpy_xor(const GFSymbol** data, GFSymbol**parity, GFSymbol**xor_result, int size, int index) {//IFFT in the proposed basis
		for (unsigned i = 0; i < size; ++i)
			memcpy(parity[i], data[i], sizeof(GFSymbol) * pack_leng);

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);
		int depart_no_4, depart_no;
		for (depart_no = 1, depart_no_4 = 4; depart_no_4 <= size; depart_no = depart_no_4, depart_no_4 <<= 2) {

			for (int j = depart_no; j <= size; j += depart_no << 2) {
				GFSymbol skew01 = skewVec[j + index - 1];
				GFSymbol skew02 = skewVec[j + depart_no + index - 1];
				GFSymbol skew23 = skewVec[j + depart_no * 2 + index - 1];
				if (depart_no_4 == size && xor_result) {
					for (int i = j - depart_no; i < j; i++) {

						const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
						const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
						const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
						const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
						const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
						const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);

						__m256i* x32 = reinterpret_cast<__m256i*>(parity[i]);
						__m256i* y32 = reinterpret_cast<__m256i*>(parity[i + depart_no]);
						__m256i* z32 = reinterpret_cast<__m256i*>(parity[i + depart_no * 2]);
						__m256i* w32 = reinterpret_cast<__m256i*>(parity[i + depart_no * 3]);

						__m256i* x32_result = reinterpret_cast<__m256i*>(xor_result[i]);
						__m256i* y32_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no]);
						__m256i* z32_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no * 2]);
						__m256i* w32_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no * 3]);
						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);
							x32 += 1;
							y32 += 1;

							y_data = _mm256_xor_si256(x_data, y_data);
							if (skew01 != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table01_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table01_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							__m256i z_data = _mm256_loadu_si256(z32);
							__m256i w_data = _mm256_loadu_si256(w32);

							z32 += 1;
							w32 += 1;
							w_data = _mm256_xor_si256(z_data, w_data);

							if (skew23 != mod) {
								__m256i lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table23_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table23_hi_y, hi);
								z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
							}

							z_data = _mm256_xor_si256(x_data, z_data);
							w_data = _mm256_xor_si256(y_data, w_data);


							if (skew02 != mod) {
								__m256i lo = _mm256_and_si256(z_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(z_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

								lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));
							}

							x_data = _mm256_xor_si256(x_data, _mm256_loadu_si256(x32_result));
							y_data = _mm256_xor_si256(y_data, _mm256_loadu_si256(y32_result));
							z_data = _mm256_xor_si256(z_data, _mm256_loadu_si256(z32_result));
							w_data = _mm256_xor_si256(w_data, _mm256_loadu_si256(w32_result));

							_mm256_storeu_si256(x32_result, x_data);
							_mm256_storeu_si256(y32_result, y_data);
							_mm256_storeu_si256(z32_result, z_data);
							_mm256_storeu_si256(w32_result, w_data);

							x32_result += 1;
							y32_result += 1;
							z32_result += 1;
							w32_result += 1;
						}
					}
				}
				else {
					for (int i = j - depart_no; i < j; i++) {

						const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
						const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
						const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
						const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
						const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
						const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);

						__m256i* x32 = reinterpret_cast<__m256i*>(parity[i]);
						__m256i* y32 = reinterpret_cast<__m256i*>(parity[i + depart_no]);
						__m256i* z32 = reinterpret_cast<__m256i*>(parity[i + depart_no * 2]);
						__m256i* w32 = reinterpret_cast<__m256i*>(parity[i + depart_no * 3]);

						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);

							y_data = _mm256_xor_si256(x_data, y_data);
							if (skew01 != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table01_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table01_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							__m256i z_data = _mm256_loadu_si256(z32);
							__m256i w_data = _mm256_loadu_si256(w32);

							w_data = _mm256_xor_si256(z_data, w_data);

							if (skew23 != mod) {
								__m256i lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table23_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table23_hi_y, hi);
								z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
							}

							z_data = _mm256_xor_si256(x_data, z_data);
							w_data = _mm256_xor_si256(y_data, w_data);

							if (skew02 != mod) {

								__m256i lo = _mm256_and_si256(z_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(z_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

								lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));
							}

							_mm256_storeu_si256(x32, x_data);
							_mm256_storeu_si256(y32, y_data);
							_mm256_storeu_si256(z32, z_data);
							_mm256_storeu_si256(w32, w_data);

							x32 += 1;
							y32 += 1;
							z32 += 1;
							w32 += 1;
						}
					}
				}
			}
		}
		if (depart_no < size) {
			GFSymbol skew = skewVec[depart_no + index - 1];
			if (xor_result) {
				if (skew == mod) {
					for (unsigned i = 0; i < depart_no; ++i) {
						__m256i* x32 = reinterpret_cast<__m256i*>(xor_result[i]);
						const __m256i* y32 = reinterpret_cast<const __m256i*>(parity[i]);
						const __m256i* z32 = reinterpret_cast<const __m256i*>(parity[i + depart_no]);
						for (int j = 0; j < pack_leng; j += 128) {
							__m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
							x0 = _mm256_xor_si256(x0, _mm256_loadu_si256(z32));
							__m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
							x1 = _mm256_xor_si256(x1, _mm256_loadu_si256(z32 + 1));
							__m256i x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2), _mm256_loadu_si256(y32 + 2));
							x2 = _mm256_xor_si256(x2, _mm256_loadu_si256(z32 + 2));
							__m256i x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3), _mm256_loadu_si256(y32 + 3));
							x3 = _mm256_xor_si256(x3, _mm256_loadu_si256(z32 + 3));
							_mm256_storeu_si256(x32, x0);
							_mm256_storeu_si256(x32 + 1, x1);
							_mm256_storeu_si256(x32 + 2, x2);
							_mm256_storeu_si256(x32 + 3, x3);
							x32 += 4, y32 += 4, z32 += 4;
						}
					}
				}
				else {
					for (int j = 0; j < depart_no; j++) {

						const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
						const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);


						__m256i* x32_input = reinterpret_cast<__m256i*>(parity[j]);
						__m256i* y32_input = reinterpret_cast<__m256i*>(parity[j + depart_no]);

						__m256i* x32_output = reinterpret_cast<__m256i*>(xor_result[j]);
						__m256i* y32_output = reinterpret_cast<__m256i*>(xor_result[j + depart_no]);

						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {

							__m256i x_data_out = _mm256_loadu_si256(x32_output);
							__m256i y_data_out = _mm256_loadu_si256(y32_output);
							__m256i x_data_in = _mm256_loadu_si256(x32_input);
							__m256i y_data_in = _mm256_loadu_si256(y32_input);
							y_data_in = _mm256_xor_si256(y_data_in, x_data_in);
							y_data_out = _mm256_xor_si256(y_data_out, y_data_in);
							_mm256_storeu_si256(y32_output, y_data_out);

							__m256i lo = _mm256_and_si256(y_data_in, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data_in, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data_in = _mm256_xor_si256(x_data_in, _mm256_xor_si256(lo, hi));

							x_data_out = _mm256_xor_si256(x_data_out, x_data_in);
							_mm256_storeu_si256(x32_output, x_data_out);

							x32_input += 1, y32_input += 1, x32_output += 1, y32_output += 1;

						}
					}
				}
			}
			else {
				if (skew == mod)
					VectorXOR(depart_no, parity + depart_no, parity);
				else
				{
					for (unsigned i = 0; i < depart_no; ++i)
					{
						const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
						const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

						__m256i* x32 = reinterpret_cast<__m256i*>(parity[i]);
						__m256i* y32 = reinterpret_cast<__m256i*>(parity[i + depart_no]);
						for (int j = 0; j < pack_leng; j += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);
							y_data = _mm256_xor_si256(y_data, x_data);
							_mm256_storeu_si256(y32, y_data);

							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

							_mm256_storeu_si256(x32, x_data);
							y32 += 1, x32 += 1;
						}
					}
				}


			}
		}


		return;
	}

	void ILCH_xor(GFSymbol**data, GFSymbol**xor_result, int size, int index) {//IFFT in the proposed basis

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);

		for (int depart_no = 1; depart_no < size; depart_no <<= 1) {
			for (int j = depart_no; j < size; j += depart_no << 1) {
				GFSymbol skew = skewVec[j + index - 1];

				const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
				const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

				for (int i = j - depart_no; i < j; i++) {
					__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
					__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);
					__m256i* x32_xor_result = reinterpret_cast<__m256i*>(xor_result[i]);
					__m256i* y32_xor_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no]);
					if (depart_no == size / 2)
					{
						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);
							__m256i x_data_xor_result = _mm256_loadu_si256(x32_xor_result);
							__m256i y_data_xor_result = _mm256_loadu_si256(y32_xor_result);

							y_data = _mm256_xor_si256(y_data, x_data);
							_mm256_storeu_si256(y32, y_data);

							if (skew != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							_mm256_storeu_si256(x32, x_data);

							x_data_xor_result = _mm256_xor_si256(x_data, x_data_xor_result);
							y_data_xor_result = _mm256_xor_si256(y_data, y_data_xor_result);

							_mm256_storeu_si256(x32_xor_result, x_data_xor_result);
							_mm256_storeu_si256(y32_xor_result, y_data_xor_result);

							x32 += 1;
							y32 += 1;
							x32_xor_result += 1;
							y32_xor_result += 1;
						}
					}
					else {
						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);

							y_data = _mm256_xor_si256(y_data, x_data);
							_mm256_storeu_si256(y32, y_data);

							if (skew != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							_mm256_storeu_si256(x32, x_data);

							x32 += 1;
							y32 += 1;
						}
					}
				}
			}
		}
		return;
	}

	void ILCH4_xor(GFSymbol**data, GFSymbol**xor_result, int size, int index) {//IFFT in the proposed basis

		const __m256i clr_mask = _mm256_set1_epi8(0x0f);
		int depart_no_4, depart_no;
		for (depart_no = 1, depart_no_4 = 4; depart_no_4 <= size; depart_no = depart_no_4, depart_no_4 <<= 2) {

			for (int j = depart_no; j <= size; j += depart_no << 2) {
				GFSymbol skew01 = skewVec[j + index - 1];
				GFSymbol skew02 = skewVec[j + depart_no + index - 1];
				GFSymbol skew23 = skewVec[j + depart_no * 2 + index - 1];
				if (depart_no_4 == size && xor_result) {
					for (int i = j - depart_no; i < j; i++) {

						const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
						const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
						const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
						const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
						const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
						const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);

						__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
						__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);
						__m256i* z32 = reinterpret_cast<__m256i*>(data[i + depart_no * 2]);
						__m256i* w32 = reinterpret_cast<__m256i*>(data[i + depart_no * 3]);

						__m256i* x32_result = reinterpret_cast<__m256i*>(xor_result[i]);
						__m256i* y32_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no]);
						__m256i* z32_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no * 2]);
						__m256i* w32_result = reinterpret_cast<__m256i*>(xor_result[i + depart_no * 3]);
						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);
							x32 += 1;
							y32 += 1;

							y_data = _mm256_xor_si256(x_data, y_data);
							if (skew01 != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table01_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table01_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							__m256i z_data = _mm256_loadu_si256(z32);
							__m256i w_data = _mm256_loadu_si256(w32);

							z32 += 1;
							w32 += 1;
							w_data = _mm256_xor_si256(z_data, w_data);

							if (skew23 != mod) {
								__m256i lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table23_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table23_hi_y, hi);
								z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
							}

							z_data = _mm256_xor_si256(x_data, z_data);
							w_data = _mm256_xor_si256(y_data, w_data);


							if (skew02 != mod) {
								__m256i lo = _mm256_and_si256(z_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(z_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

								lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));
							}

							x_data = _mm256_xor_si256(x_data, _mm256_loadu_si256(x32_result));
							y_data = _mm256_xor_si256(y_data, _mm256_loadu_si256(y32_result));
							z_data = _mm256_xor_si256(z_data, _mm256_loadu_si256(z32_result));
							w_data = _mm256_xor_si256(w_data, _mm256_loadu_si256(w32_result));

							_mm256_storeu_si256(x32_result, x_data);
							_mm256_storeu_si256(y32_result, y_data);
							_mm256_storeu_si256(z32_result, z_data);
							_mm256_storeu_si256(w32_result, w_data);

							x32_result += 1;
							y32_result += 1;
							z32_result += 1;
							w32_result += 1;
						}
					}
				}
				else {
					for (int i = j - depart_no; i < j; i++) {

						const __m256i table01_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[0]);
						const __m256i table01_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew01].Value[1]);
						const __m256i table02_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[0]);
						const __m256i table02_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew02].Value[1]);
						const __m256i table23_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[0]);
						const __m256i table23_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew23].Value[1]);

						__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
						__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);
						__m256i* z32 = reinterpret_cast<__m256i*>(data[i + depart_no * 2]);
						__m256i* w32 = reinterpret_cast<__m256i*>(data[i + depart_no * 3]);

						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);

							y_data = _mm256_xor_si256(x_data, y_data);
							if (skew01 != mod) {
								__m256i lo = _mm256_and_si256(y_data, clr_mask);
								lo = _mm256_shuffle_epi8(table01_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(y_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table01_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));
							}
							__m256i z_data = _mm256_loadu_si256(z32);
							__m256i w_data = _mm256_loadu_si256(w32);

							w_data = _mm256_xor_si256(z_data, w_data);

							if (skew23 != mod) {
								__m256i lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table23_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table23_hi_y, hi);
								z_data = _mm256_xor_si256(z_data, _mm256_xor_si256(lo, hi));
							}

							z_data = _mm256_xor_si256(x_data, z_data);
							w_data = _mm256_xor_si256(y_data, w_data);

							if (skew02 != mod) {

								__m256i lo = _mm256_and_si256(z_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								__m256i hi = _mm256_srli_epi64(z_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

								lo = _mm256_and_si256(w_data, clr_mask);
								lo = _mm256_shuffle_epi8(table02_lo_y, lo);
								hi = _mm256_srli_epi64(w_data, 4);
								hi = _mm256_and_si256(hi, clr_mask);
								hi = _mm256_shuffle_epi8(table02_hi_y, hi);
								y_data = _mm256_xor_si256(y_data, _mm256_xor_si256(lo, hi));
							}

							_mm256_storeu_si256(x32, x_data);
							_mm256_storeu_si256(y32, y_data);
							_mm256_storeu_si256(z32, z_data);
							_mm256_storeu_si256(w32, w_data);

							x32 += 1;
							y32 += 1;
							z32 += 1;
							w32 += 1;
						}
					}
				}
			}
		}
		if (depart_no < size) {
			GFSymbol skew = skewVec[depart_no + index - 1];
			if (xor_result) {
				if (skew == mod) {
					for (unsigned i = 0; i < depart_no; ++i) {
						__m256i* x32 = reinterpret_cast<__m256i*>(xor_result[i]);
						const __m256i* y32 = reinterpret_cast<const __m256i*>(data[i]);
						const __m256i* z32 = reinterpret_cast<const __m256i*>(data[i + depart_no]);
						for (int j = 0; j < pack_leng; j += 128) {
							__m256i x0 = _mm256_xor_si256(_mm256_loadu_si256(x32), _mm256_loadu_si256(y32));
							x0 = _mm256_xor_si256(x0, _mm256_loadu_si256(z32));
							__m256i x1 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 1), _mm256_loadu_si256(y32 + 1));
							x1 = _mm256_xor_si256(x1, _mm256_loadu_si256(z32 + 1));
							__m256i x2 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 2), _mm256_loadu_si256(y32 + 2));
							x2 = _mm256_xor_si256(x2, _mm256_loadu_si256(z32 + 2));
							__m256i x3 = _mm256_xor_si256(_mm256_loadu_si256(x32 + 3), _mm256_loadu_si256(y32 + 3));
							x3 = _mm256_xor_si256(x3, _mm256_loadu_si256(z32 + 3));
							_mm256_storeu_si256(x32, x0);
							_mm256_storeu_si256(x32 + 1, x1);
							_mm256_storeu_si256(x32 + 2, x2);
							_mm256_storeu_si256(x32 + 3, x3);
							x32 += 4, y32 += 4, z32 += 4;
						}
					}
				}
				else {
					for (int j = 0; j < depart_no; j++) {

						const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
						const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);


						__m256i* x32_input = reinterpret_cast<__m256i*>(data[j]);
						__m256i* y32_input = reinterpret_cast<__m256i*>(data[j + depart_no]);

						__m256i* x32_output = reinterpret_cast<__m256i*>(xor_result[j]);
						__m256i* y32_output = reinterpret_cast<__m256i*>(xor_result[j + depart_no]);

						for (int p_leng = 0; p_leng < pack_leng; p_leng += 32) {

							__m256i x_data_out = _mm256_loadu_si256(x32_output);
							__m256i y_data_out = _mm256_loadu_si256(y32_output);
							__m256i x_data_in = _mm256_loadu_si256(x32_input);
							__m256i y_data_in = _mm256_loadu_si256(y32_input);
							y_data_in = _mm256_xor_si256(y_data_in, x_data_in);
							y_data_out = _mm256_xor_si256(y_data_out, y_data_in);
							_mm256_storeu_si256(y32_output, y_data_out);

							__m256i lo = _mm256_and_si256(y_data_in, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data_in, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data_in = _mm256_xor_si256(x_data_in, _mm256_xor_si256(lo, hi));

							x_data_out = _mm256_xor_si256(x_data_out, x_data_in);
							_mm256_storeu_si256(x32_output, x_data_out);

							x32_input += 1, y32_input += 1, x32_output += 1, y32_output += 1;

						}
					}
				}
			}
			else {
				if (skew == mod)
					VectorXOR(depart_no, data + depart_no, data);
				else
				{
					for (unsigned i = 0; i < depart_no; ++i)
					{
						const __m256i table_lo_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[0]);
						const __m256i table_hi_y = _mm256_loadu_si256(&Multiply256LUT[skew].Value[1]);

						__m256i* x32 = reinterpret_cast<__m256i*>(data[i]);
						__m256i* y32 = reinterpret_cast<__m256i*>(data[i + depart_no]);
						for (int j = 0; j < pack_leng; j += 32) {
							__m256i x_data = _mm256_loadu_si256(x32);
							__m256i y_data = _mm256_loadu_si256(y32);
							y_data = _mm256_xor_si256(y_data, x_data);
							_mm256_storeu_si256(y32, y_data);

							__m256i lo = _mm256_and_si256(y_data, clr_mask);
							lo = _mm256_shuffle_epi8(table_lo_y, lo);
							__m256i hi = _mm256_srli_epi64(y_data, 4);
							hi = _mm256_and_si256(hi, clr_mask);
							hi = _mm256_shuffle_epi8(table_hi_y, hi);
							x_data = _mm256_xor_si256(x_data, _mm256_xor_si256(lo, hi));

							_mm256_storeu_si256(x32, x_data);
							y32 += 1, x32 += 1;
						}
					}
				}
			}
		}

	}
	void formal_derivative(GFSymbol* cos, int size) {//formal derivative of polynomial in the new basis
		for (int i = 0; i < size; i++) {
			cos[i] = mulE(cos[i], B[i]);
		}

		for (int i = 1; i < size; i++) {
			cos[i - 1] = 0;
			int leng = ((i ^ i - 1) + 1) >> 1;
			for (int j = i - leng; j < i; j++)
				cos[j] ^= cos[j + leng];
		}
		cos[size - 1] = 0;

		for (int i = 0; i < size; i++) {
			cos[i] = mulE(cos[i], mod - B[i]);
		}
		return;
	}
	void IFLT(GFSymbol* data, int size, int index) {//IFFT in the proposed basis
		for (int depart_no = 1; depart_no < size; depart_no <<= 1) {

			for (int j = depart_no; j < size; j += depart_no << 1) {
				for (int i = j - depart_no; i < j; i++)
					data[i + depart_no] ^= data[i];

				GFSymbol skew = skewVec[j + index - 1];

				if (skew != mod)
					for (int i = j - depart_no; i < j; i++)
						data[i] ^= mulE(data[i + depart_no], skew);
			}
		}
		return;
	}

	void FLT(GFSymbol* data, int size, int index) {//FFT in the proposed basis
		for (int depart_no = size >> 1; depart_no > 0; depart_no >>= 1) {
			for (int j = depart_no; j < size; j += depart_no << 1) {
				GFSymbol skew = skewVec[j + index - 1];
				if (skew != mod)
					for (int i = j - depart_no; i < j; i++)
						data[i] ^= mulE(data[i + depart_no], skew);
				for (int i = j - depart_no; i < j; i++)
					data[i + depart_no] ^= data[i];
			}
		}
		return;
	}
	void ReedSolomonEncodeL(const GFSymbol** data, GFSymbol** parity) {

		//IFLT_cpy(data,parity, k, 0);//parity<-IFFT(data,k,0)
		ILCH4_cpy(data, parity, k, 0);
		for (int i = 2 * k; i < Size; i += k) {
			for (int j = 0; j < k; j++) {
				memcpy(parity[j + i - k], parity[j], sizeof(GFSymbol) * pack_leng);
			}
			LCH4(&parity[i - k], k, i);
		}
		LCH4(&parity[0], k, k);

		return;
	}
	void ReedSolomonEncodeH(const GFSymbol** data, GFSymbol** parity)
	{
		ILCH4_cpy(data, parity, t, t);//parity<-IFFT(data,t,t)

		for (int i = 2 * t; i < Size; i += t) {
			ILCH4_cpy_xor(&data[i - t], &parity[t], parity, t, i);
		}
		LCH4(parity, t, 0);

		return;
	}

	namespace Algorithm1 {

		void ReedSolomondecodeL(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2) {
			//代码里的多项式都是点值。
			for (int i = 0; i < Size; i++)
				//R
				log_walsh2[i] = erasure[i];//x是擦除位置，R(x)=1，否则R(x)=0      

			walsh(log_walsh2, Size);//WHT(R(x))

			for (int i = 0; i < Size; i++)
				
				log_walsh2[i] = (unsigned long)log_walsh2[i] * log_walsh[i] % mod;//WHT(R)*WHT(Log)

			walsh(log_walsh2, Size); ////Log(L(x))=(R*Log)(x)=WHT(WHT(R)*WHT(Log))   用R(x)标记*某一个式子来实现删除位置的确定？
			//mul_mem函数是a*alpha(b)?
			for (int i = 0; i < k; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], data[i], log_walsh2[i], pack_leng);//g(aj)=f(aj)*L(aj)  此处log_walsh2是删除位置多项式L(x)
			}

			for (int i = k; i < Size; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], parity[i - k], log_walsh2[i],  pack_leng);
			}

			ILCH4(codeword, Size, 0);

			formal_derivative(codeword, Size);

			LCH4(codeword, Size, 0);

			for (int i = 0; i < k; ++i)
				if (erasure[i])
					mul_mem(codeword[i], codeword[i], mod - log_walsh2[i],  pack_leng);
		}

		void ReedSolomondecodeH(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2) {

			for (int i = 0; i < Size; i++)
				log_walsh2[i] = erasure[i];

			walsh(log_walsh2, Size);

			for (int i = 0; i < Size; i++)
				log_walsh2[i] = (unsigned long)log_walsh2[i] * log_walsh[i] % mod;

			walsh(log_walsh2, Size);

			for (int i = 0; i < t; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], parity[i], log_walsh2[i],  pack_leng);
			}

			for (int i = t; i < Size; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], data[i - t], log_walsh2[i],  pack_leng);
			}

			ILCH4(codeword, Size, 0);

			formal_derivative(codeword, Size);

			LCH4(codeword, Size, 0);

			for (int i = t; i < Size; ++i) 
				if (erasure[i])
					mul_mem(codeword[i], codeword[i], mod - log_walsh2[i], pack_leng);
		}

	}
	namespace Algorithm2 {

		void ReedSolomondecodeL(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2) {

			for (int i = 0; i < Size; i++)
				log_walsh2[i] = erasure[i];

			walsh(log_walsh2, Size);

			for (int i = 0; i < Size; i++)
				log_walsh2[i] = (unsigned long)log_walsh2[i] * log_walsh[i] % mod;

			walsh(log_walsh2, Size);

			for (int i = 0; i < k; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], data[i], log_walsh2[i],  pack_leng);
			}

			for (int i = k; i < Size; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], parity[i - k], log_walsh2[i], pack_leng);
			}

			for (int i = 0; i < Size; i += k)
			{
				ILCH4(&codeword[i], k, i);
			}

			formal_derivative(&codeword[0], k);

			for (int j = 1; j < Size / k; j++)
			{
				for (int i = 0; i < k; i++)
				{
					mul_mem(codeword[j * k + i], codeword[j * k + i], coefL[j - 1], pack_leng);
					xor_mem(codeword[i], codeword[j * k + i], pack_leng);
				}
			}

			LCH4(&codeword[0], k, 0);

			for (int i = 0; i < k; i++) {
				if (erasure[i])
					mul_mem(codeword[i], codeword[i], mod - log_walsh2[i],  pack_leng);
			}
		}

	}

	namespace Algorithm3 {

		void ReedSolomondecodeH(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2)//高码率译码
		{
			for (int i = 0; i < Size; i++)
				log_walsh2[i] = erasure[i];

			walsh(log_walsh2, Size);

			for (int i = 0; i < Size; i++)
				log_walsh2[i] = (unsigned long)log_walsh2[i] * log_walsh[i] % mod;

			walsh(log_walsh2, Size);

			for (int i = 0; i < t; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					memcpy(codeword[i], parity[i], sizeof(GFSymbol) * pack_leng);
			}

			for (int i = t; i < Size; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					memcpy(codeword[i], data[i - t], sizeof(GFSymbol) * pack_leng);
			}

			ILCH4(codeword, t, 0);

			for (int i = t; i < Size; i += t)
			{
				ILCH4_xor(codeword + i, codeword, t, i);
			}

			LCH4(codeword, t, 0);//求伴随多项式点值

			for (int i = 0; i < t; i++) {
				if (erasure[i])
					memset(codeword[i], 0, sizeof(GFSymbol) * pack_leng);
				else
					mul_mem(codeword[i], codeword[i], log_walsh2[i], pack_leng);
			}

			ILCH4(codeword, t, 0);//求Z(x)的系数

			for (int j = t; j < Size; j += t)
			{
				for (int i = 0; i < t; i++) {
					memcpy(codeword[j + i], codeword[i], sizeof(GFSymbol) * pack_leng);
				}

				LCH4(&codeword[j], t, j);//求Z(x)在信息位上的点值

				for (int i = 0; i < t; i++)
					if (erasure[j + i])//只找擦除位置
					{
						mul_mem(codeword[j + i], codeword[j + i], coefH[j + i],  pack_leng);//求擦除值
						mul_mem(codeword[j + i], codeword[j + i], mod - log_walsh2[j + i],  pack_leng);
					}
			}
		}

	}
}