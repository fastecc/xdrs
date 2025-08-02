#include"P_function.h"
#define A1//算法
//#define CHECK//是否验证译码正确性
#include <math.h>
#include <cmath>
#include<thread>

using namespace function;

struct test_params {//GF(256)
	//L:k=pow(2,i),k:0<->128  
	//H:t=pow(2,i),k:128<->256 
	static const unsigned pack_leng = 1024;
	static const unsigned Trials = 1000;
	static const unsigned k = 8;
	unsigned logk = log2(k);
	static const unsigned t = Size - k;
	static const unsigned seed = 2;

};

static void Benchmark(test_params& params) {

	unsigned parity_count = params.t + (params.k >= params.t ? params.t : 0);

	GFSymbol* index_of = (GFSymbol*)calloc(Size, sizeof(GFSymbol));

	GFSymbol* alpha_to = (GFSymbol*)calloc(Size, sizeof(GFSymbol));

	GFSymbol* skewVec = (GFSymbol*)calloc(mod, sizeof(GFSymbol));

	GFSymbol* B = (GFSymbol*)calloc(Size, sizeof(GFSymbol));

	GFSymbol* log_walsh = (GFSymbol*)calloc(Size, sizeof(GFSymbol));

	GFSymbol* base = (GFSymbol*)calloc(len - 1, sizeof(GFSymbol));

	GFSymbol* coefL = (GFSymbol*)calloc(Size / params.k - 1, sizeof(GFSymbol));

	GFSymbol* coefH = (GFSymbol*)calloc(Size, sizeof(GFSymbol));

	int* s = (int*)calloc(len, sizeof(int));
	
    main_params(params.pack_leng, params.Trials, params.k);//传递数据参数

	array_params(index_of, alpha_to, skewVec, B, log_walsh, base, coefL, coefH, s);//传递数组参数

	init();//fill index_of table and alpha_to table

	init_dec();//initialize skewVec[], B[], index_of_walsh[]

	vector<GFSymbol*> original_data(params.k);

	vector<GFSymbol*> parity(parity_count);

	vector<GFSymbol*> codeword(Size);

	FunctionTimer time_encode("encode");

	FunctionTimer time_decode("decode");

	for (int trial = 0; trial < params.Trials; trial++) {
		/*-------------------------------------------------------------------------------------------------------------------*/
				// Allocate memory:
		for (unsigned i = 0, count = params.k; i < count; ++i)
			original_data[i] = (GFSymbol*)calloc(params.pack_leng, sizeof(GFSymbol));//消息
		for (unsigned i = 0, count = parity_count; i < count; ++i)
			parity[i] = (GFSymbol*)calloc(params.pack_leng, sizeof(GFSymbol));//校验位
		for (unsigned i = 0, count = Size; i < count; ++i)
			codeword[i] = (GFSymbol*)calloc(params.pack_leng, sizeof(GFSymbol));//译码结果
		
		GFSymbol* fL = (GFSymbol*)calloc(2 * params.k * (params.logk + 1) + Size, sizeof(GFSymbol));
		vector<GFSymbol*> SIMD_fL(2 * params.k * (params.logk + 1) + Size);
		vector<GFSymbol*> temp_vec(params.k);
		for (unsigned i = 0, count = 2 * params.k * (params.logk + 1) + Size; i < count; ++i)
			SIMD_fL[i] = (GFSymbol*)calloc(params.pack_leng, sizeof(GFSymbol));//SIMD_fL
		for (unsigned i = 0, count = params.k; i < count; ++i)
			temp_vec[i] = (GFSymbol*)calloc(params.pack_leng, sizeof(GFSymbol));//temp_vec
/*-------------------------------------------------------------------------------------------------------------------*/
		//erasure simulation:

		bool* erasure = (bool*)calloc(Size, sizeof(bool));//Array indicating erasures

		for (int i = 0; i < params.k; i++)
			erasure[i] = 0;
		for (int i = params.k; i < Size; i++)
			erasure[i] = 1;
		for (int i = Size - 1; i > 0; i--) {//permuting the erasure array
			int pos = rand() % (i + 1);
			if (i != pos) {
				bool tmp = erasure[i];
				erasure[i] = erasure[pos];
				erasure[pos] = tmp;
			}
		}

		//Erasure decoding:

		GFSymbol* log_walsh2 = (GFSymbol*)calloc(Size, sizeof(GFSymbol));
		/*-------------------------------------------------------------------------------------------------------------------*/
		// Generate data:

		PCGRandom prng;
		prng.Seed(params.seed, trial);

		for (unsigned i = 0; i < params.k; ++i) {
			WriteRandomSelfCheckingPacket(prng, original_data[i], params.pack_leng);
			//memset(original_data[i], i, sizeof(GFSymbol) * params.pack_leng);
		}
		/*-------------------------------------------------------------------------------------------------------------------*/
		time_encode.BeginCall();//编码计时

		// Encode:

#ifndef A3
		if (params.k <= Size / 2) 
			ReedSolomonEncodeL((const GFSymbol**)&original_data[0], (GFSymbol**)&parity[0]);//改进版本
		else
			ReedSolomonEncodeH((const GFSymbol**)&original_data[0], (GFSymbol**)&parity[0]);
#else
		ReedSolomonEncodeH((const GFSymbol**)&original_data[0], (GFSymbol**)&parity[0]);
#endif 

		time_encode.EndCall();//编码计时
/*-------------------------------------------------------------------------------------------------------------------*/

		time_decode.BeginCall();//译码计时

		//Decode
#ifdef A1
		// A1-Decode:

		if (params.k <= Size / 2)
			Algorithm1::ReedSolomondecodeL((const GFSymbol**)&original_data[0], (const GFSymbol**)&parity[0], (GFSymbol**)&codeword[0], erasure, log_walsh2);
		else
			Algorithm1::ReedSolomondecodeH((const GFSymbol**)&original_data[0], (const GFSymbol**)&parity[0], (GFSymbol**)&codeword[0], erasure, log_walsh2);
#endif
#ifdef A2
		// A2-Decode:
		Algorithm2::ReedSolomondecodeL((const GFSymbol**)&original_data[0], (const GFSymbol**)&parity[0], (GFSymbol**)&codeword[0], erasure, log_walsh2);
#endif
#ifdef A3
		// A3-Decode:

		Algorithm3::ReedSolomondecodeH((const GFSymbol**)&original_data[0], (const GFSymbol**)&parity[0], (GFSymbol**)&codeword[0], erasure, log_walsh2);
#endif


		time_decode.EndCall();//译码计时
/*-------------------------------------------------------------------------------------------------------------------*/

#ifdef CHECK
#ifndef A3
		// Check:
		if (params.k <= Size / 2) {
			//低码率
			for (int l = 0; l < params.pack_leng; l++) {
				for (int i = 0; i < params.k; i++) {//Check the correctness of the result
					if (erasure[i] == 1)
						if (original_data[i][l] != codeword[i][l]) {
							printf("Decoding Error!\n");
						}
				}
				printf("Decoding is successful!\n");
			}
		}
		else {
			//高码率
			for (int l = 0; l < params.pack_leng; l++) {
				for (int i = params.t; i < Size; i++) {//Check the correctness of the result
					if (erasure[i] == 1)
						if (original_data[i - params.t][l] != codeword[i][l]) {
							printf("Decoding Error!\n");
						}
				}
				printf("Decoding is successful!\n");
			}
		}
#else
	//高码率
		for (int l = 0; l < params.pack_leng; l++) {
			for (int i = params.t; i < Size; i++) {//Check the correctness of the result
				if (erasure[i] == 1)
					if (original_data[i - params.t][l] != codeword[i][l]) {
						printf("Decoding Error!\n");
					}
			}
			printf("Decoding is successful!\n");
			}
#endif 
#endif 

		/*-------------------------------------------------------------------------------------------------------------------*/

				// Free memory:
		
		free(fL);
		for (unsigned i = 0, count = 2 * params.k * (params.logk + 1) + Size; i < count; ++i)
			free(SIMD_fL[i]);
		for (unsigned i = 0, count = params.k; i < count; ++i)
			free(temp_vec[i]);
		for (unsigned i = 0, count = params.k; i < count; ++i) 
			free(original_data[i]);
		for (unsigned i = 0, count = parity_count; i < count; ++i)
			free(parity[i]);
		for (unsigned i = 0, count = Size; i < count; ++i)
			free(codeword[i]);
		free(erasure);
		free(log_walsh2);


		/*-------------------------------------------------------------------------------------------------------------------*/

		}



	free(index_of);
	free(alpha_to);
	free(skewVec);
	free(B);
	free(log_walsh);
	free(base);
	free(s);
	/*-------------------------------------------------------------------------------------------------------------------*/



	float Encode_Input = (uint64_t)params.pack_leng * params.k * params.Trials / (float)(time_encode.TotalUsec);

	float Encode_Output = (uint64_t)params.pack_leng * params.t * params.Trials / (float)(time_encode.TotalUsec);

	float Decode_Input = (uint64_t)params.pack_leng * params.k * params.Trials / (float)(time_decode.TotalUsec);

	float Decode_Output = (uint64_t)params.pack_leng * (params.k - params.k * params.k / Size) * params.Trials / (float)(time_decode.TotalUsec);

	cout << " Encoder(" << (uint64_t)params.pack_leng * params.k / 1000000.f << " MB in " << params.k << " pieces, " << params.t << " losses): Input=" << Encode_Input << " MB/s, Output=" << Encode_Output << " MB/s" << endl;

	cout << " Decoder(" << (uint64_t)params.pack_leng * params.k / 1000000.f << " MB in " << params.k << " pieces, " << params.t << " losses): Input=" << Decode_Input << " MB/s, Output=" << Decode_Output << " MB/s" << endl << endl;

	/*-------------------------------------------------------------------------------------------------------------------*/
	float x = (uint64_t)params.pack_leng * params.k * params.Trials / (float)(time_decode.TotalUsec);
	cout <<"T=" << time_decode.TotalUsec << endl;

	}

int main()
{
	test_params params;

	Benchmark(params);

	return 1;
}