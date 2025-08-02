#pragma once
#include<immintrin.h>
#include<stdint.h>
#include <memory>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

typedef unsigned char GFSymbol;
static const GFSymbol mask = 0x1D; //GF(2^8): x^8 + x^4 + x^3 + x^2 + 1
static const GFSymbol CantorBase[] = { 1, 214, 152, 146, 86, 200, 88, 230 };//Cantor basis
static const unsigned len = 8;
static const unsigned Size = 1 << len;
static const unsigned mod = Size - 1;

//------------------------------------------------------------------------------
// Timing
//------------------------------------------------------------------------------
// Windows

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN

#ifndef _WINSOCKAPI_
#define DID_DEFINE_WINSOCKAPI
#define _WINSOCKAPI_
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 /* Windows 7+ */
#endif

#include <windows.h>
#endif

#ifdef DID_DEFINE_WINSOCKAPI
#undef _WINSOCKAPI_
#undef DID_DEFINE_WINSOCKAPI
#endif
#ifndef _WIN32
#include <sys/time.h>
#endif

static uint64_t GetTimeUsec()
{
#ifdef _WIN32
	LARGE_INTEGER timeStamp = {};
	if (!::QueryPerformanceCounter(&timeStamp))
		return 0;
	static double PerfFrequencyInverse = 0.;
	if (PerfFrequencyInverse == 0.)
	{
		LARGE_INTEGER freq = {};
		if (!::QueryPerformanceFrequency(&freq) || freq.QuadPart == 0)
			return 0;
		PerfFrequencyInverse = 1000000. / (double)freq.QuadPart;
	}
	return (uint64_t)(PerfFrequencyInverse * timeStamp.QuadPart);
#else
	struct timeval tv;
	gettimeofday(&tv, nullptr);
	return 1000000 * tv.tv_sec + tv.tv_usec;
#endif // _WIN32
}
#define LEO_DEBUG_ASSERT(cond) ;

//------------------------------------------------------------------------------
// FunctionTimer

class FunctionTimer
{
public:
	FunctionTimer(const std::string& name)
	{
		FunctionName = name;
	}
	void BeginCall()
	{
		LEO_DEBUG_ASSERT(t0 == 0);
		t0 = GetTimeUsec();
	}
	void EndCall()
	{
		LEO_DEBUG_ASSERT(t0 != 0);
		const uint64_t t1 = GetTimeUsec();
		const uint64_t delta = t1 - t0;
		if (++Invokations == 1)
			MaxCallUsec = MinCallUsec = delta;
		else if (MaxCallUsec < delta)
			MaxCallUsec = delta;
		else if (MinCallUsec > delta)
			MinCallUsec = delta;
		TotalUsec += delta;
		t0 = 0;
	}
	void Reset()
	{
		LEO_DEBUG_ASSERT(t0 == 0);
		t0 = 0;
		Invokations = 0;
		TotalUsec = 0;
	}

	uint64_t t0 = 0;
	uint64_t Invokations = 0;
	uint64_t TotalUsec = 0;
	uint64_t MaxCallUsec = 0;
	uint64_t MinCallUsec = 0;
	std::string FunctionName;
};

//------------------------------------------------------------------------------
// PCG PRNG
// From http://www.pcg-random.org/

class PCGRandom//产生随机数
{
public:
	inline void Seed(uint64_t y, uint64_t x = 0)
	{
		State = 0;
		Inc = (y << 1u) | 1u;
		Next();
		State += x;
		Next();
	}

	inline uint32_t Next()
	{
		const uint64_t oldstate = State;
		State = oldstate * UINT64_C(6364136223846793005) + Inc;
		const uint32_t xorshifted = (uint32_t)(((oldstate >> 18) ^ oldstate) >> 27);
		const uint32_t rot = oldstate >> 59;
		return (xorshifted >> rot) | (xorshifted << ((uint32_t)(-(int32_t)rot) & 31));
	}

	uint64_t State = 0, Inc = 0;
};

//------------------------------------------------------------------------------
// Self-Checking Packet

static void WriteRandomSelfCheckingPacket(PCGRandom& prng, void* packet, unsigned bytes)
{
	uint8_t* buffer = (uint8_t*)packet;
#ifdef TEST_DATA_ALL_SAME
	if (bytes != 0)
#else
	if (bytes < 16)
#endif
	{
		LEO_DEBUG_ASSERT(bytes >= 2);
		buffer[0] = (uint8_t)prng.Next();
		for (unsigned i = 1; i < bytes; ++i)
		{
			buffer[i] = buffer[0];
		}
	}
	else
	{
		uint32_t crc = bytes;
		*(uint32_t*)(buffer + 4) = bytes;
		for (unsigned i = 8; i < bytes; ++i)
		{
			uint8_t v = (uint8_t)prng.Next();
			buffer[i] = v;
			crc = (crc << 3) | (crc >> (32 - 3));
			crc += v;
		}
		*(uint32_t*)buffer = crc;
	}
}

static bool CheckPacket(const void* packet, unsigned bytes)
{
	uint8_t* buffer = (uint8_t*)packet;
#ifdef TEST_DATA_ALL_SAME
	if (bytes != 0)
#else
	if (bytes < 16)
#endif
	{
		if (bytes < 2)
			return false;

		uint8_t v = buffer[0];
		for (unsigned i = 1; i < bytes; ++i)
		{
			if (buffer[i] != v)
				return false;
		}
	}
	else
	{
		uint32_t crc = bytes;
		uint32_t readBytes = *(uint32_t*)(buffer + 4);
		if (readBytes != bytes)
			return false;
		for (unsigned i = 8; i < bytes; ++i)
		{
			uint8_t v = buffer[i];
			crc = (crc << 3) | (crc >> (32 - 3));
			crc += v;
		}
		uint32_t readCRC = *(uint32_t*)buffer;
		if (readCRC != crc)
			return false;
	}
	return true;
}


namespace function {

	void array_params(GFSymbol* index_of, GFSymbol* alpha_to,
		GFSymbol* skewVec, GFSymbol* B,
		GFSymbol* log_walsh, GFSymbol* base,
		GFSymbol* coefL, GFSymbol* coefH,
		int* s);//传递数据参数

	void main_params(unsigned pack_leng, unsigned Trails, unsigned k);//传递数组参数

	GFSymbol mulE(GFSymbol a, GFSymbol b);

	void init();//初始化指数表、对数表以及乘法表

	void init_dec();//初始化skewVec[], B[], index_of_walsh[]，coefL[]，coefH[]

	void walsh(GFSymbol* data, int size);//walsh变换

	static void mul_mem(GFSymbol* x, const GFSymbol* y, GFSymbol log_m, uint64_t bytes);//x[]=y[]*log_m，bytes为个数，并非字节数

	void xor_mem(GFSymbol* x, const GFSymbol* y, uint64_t bytes);//x[]^=y[]，bytes为个数，并非字节数

	void xor_mem4(
		void* vx_0, const void* vy_0,
		void* vx_1, const void* vy_1,
		void* vx_2, const void* vy_2,
		void* vx_3, const void* vy_3,
		uint64_t bytes);

	void VectorXOR(
		unsigned count,
		GFSymbol** x,
		GFSymbol** y);

	void formal_derivative(GFSymbol* cos, int size);//形式导数

	void formal_derivative(GFSymbol** cos, int size);//SIMD形式导数

	void LCH(GFSymbol** data, int size, int index);//原位2点FFT

	void LCH4(GFSymbol** data, int size, int index);//原位4点FFT

	void ILCH(GFSymbol** data, int size, int index);//原位2点IFFT

	void ILCH4(GFSymbol** data, int size, int index);//原位4点IFFT

	void ILCH_cpy(const GFSymbol** data, GFSymbol**parity, int size, int index);//赋值2点IFFT

	void ILCH4_cpy(const GFSymbol** data, GFSymbol**parity, int size, int index);//赋值4点IFFT

	void ILCH_cpy_xor(const GFSymbol** data, GFSymbol**parity, GFSymbol**xor_result, int size, int index);//赋值异或2点IFFT

	void ILCH4_cpy_xor(const GFSymbol** data, GFSymbol**parity, GFSymbol**xor_result, int size, int index);//赋值异或4点IFFT

	void ILCH_xor(const GFSymbol** data, GFSymbol**xor_result, int size, int index);//异或2点IFFT

	void ILCH4_xor(const GFSymbol** data, GFSymbol**xor_result, int size, int index);//异或4点IFFT

	void ReedSolomonEncodeL(const GFSymbol** data, GFSymbol** parity);//低码率编码

	void ReedSolomonEncodeH(const GFSymbol** data, GFSymbol** parity);//高码率编码

	namespace Algorithm1 {

		void ReedSolomondecodeL(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2);//算法1低码率译码

		void ReedSolomondecodeH(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2);//算法1高码率译码
	}
	namespace Algorithm2 {

		void ReedSolomondecodeL(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2);//算法2低码率译码
	}
	namespace Algorithm3 {

		void ReedSolomondecodeH(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure, GFSymbol* log_walsh2); //算法3高码率译码
	}
	namespace Algorithm4 {

		void ReedSolomondecodeL(const GFSymbol** data, const GFSymbol** parity, GFSymbol** codeword, bool* erasure,GFSymbol* fL, GFSymbol** SIMD_fL, GFSymbol** temp_vec, unsigned logk); //算法4低码率译码
	
	}

}


