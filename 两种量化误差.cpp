#include <bits/stdc++.h>
using namespace std;

const int Num = 100;
const int Bits = 8;
const float Mean = 500.0;
const float StdDev = 43232.0f;
float P = -1; // 此处若设置 P 为 -1 则按 1/error 加权



vector<float> Get_Data(float &Min_x,float &Max_x,map<float,int> &data_num, map<float,float> &data_f,float mean = 0.0f, float stddev = 1.0f) {
	// 设置随机数引擎和正态分布器
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stddev);
	// 存储生成的随机数
	std::vector<float> data;
	// 生成指定数量的随机数
	for (int i = 0; i < Num; ++i) {
		float value = distribution(generator);
		data.push_back(value);
		data_num[value] ++;
		Min_x = min(Min_x,value);
		Max_x = max(Max_x,value);
	}
	for(auto x : data_num) {
		float f = x.second * 1.0 / Num;
		data_f[x.first] = f;
	}

	return data;
}

float Cacl_Relative_Squ_Quant(vector<float> data,vector<int> A) {
	float RSQE = 0; // 相对量化误差
	for(auto x : data) {
		int A_R_index = lower_bound(A.begin(),A.end(),x) - A.begin(); // 找到第一个大于等于 x 的边界  即为右边界Ai，
		if(A_R_index == 0) A_R_index = 1; // 防止越界
		if(A_R_index == A.size()) A_R_index = A.size()-1; // 防止越界
		float e = ( -1 + (1.0*(A[A_R_index] + A[A_R_index-1])) / x + ((1.0)*(A[A_R_index-1] * A[A_R_index])) / (x * x) );
		RSQE += e;
	}
	cout<<"在当前量化边界下，data的相对平方量化误差为RSQE："<< RSQE <<endl;
	return RSQE;
}

float Cacl_Absolute_Squ_Quant(vector<float> data, vector<int> A, bool point = false, float p_x = 0.0) {
	float SQE = 0; // 绝对量化误差
	if(point == false) {
		for(auto x : data) {
			int A_R_index = lower_bound(A.begin(),A.end(),x) - A.begin(); // 找到第一个大于等于 x 的边界  即为右边界Ai，
			if(A_R_index == 0) A_R_index = 1; // 防止越界
			if(A_R_index == A.size()) A_R_index = A.size()-1; // 防止越界
			float e = -1 * (x * x) + (A[A_R_index - 1] + A[A_R_index]) * x - (A[A_R_index-1] * A[A_R_index]) ;
			SQE += e;
		}
		cout<<"在当前量化边界下，data的绝对平方量化误差为SQE："<< SQE <<endl;
		return SQE;
	} else {
		int A_R_index = lower_bound(A.begin(),A.end(),p_x) - A.begin(); // 找到第一个大于等于 x 的边界  即为右边界Ai，
		if(A_R_index == 0) A_R_index = 1; // 防止越界
		if(A_R_index == A.size()) A_R_index = A.size()-1; // 防止越界
		float e = -1 * (p_x * p_x) + (A[A_R_index - 1] + A[A_R_index]) * p_x - (A[A_R_index-1] * A[A_R_index]) ;
		return e;
	}

}

float Cacl_Weight_Squ_Quant(vector<float> data,vector<int> A) {
	float WSQE = 0; // 权重量化误差
	for(auto x : data) {
		int A_R_index = lower_bound(A.begin(),A.end(),x) - A.begin(); // 找到第一个大于等于 x 的边界  即为右边界Ai，
		if(A_R_index == 0) A_R_index = 1; // 防止越界
		if(A_R_index == A.size()) A_R_index = A.size()-1; // 防止越界
		float Relative_Error = ( -1 + (1.0*(A[A_R_index] + A[A_R_index-1])) / x + ((1.0)*(A[A_R_index-1] * A[A_R_index])) / (x * x) );
		float Absolute_Error = -1 * (x * x) + (A[A_R_index - 1] + A[A_R_index]) * x - (A[A_R_index-1] * A[A_R_index]) ;
		if(P == -1) P = 1.0 / Absolute_Error;
		WSQE = P*Absolute_Error + (1-P) * Relative_Error;
	}
	cout<<"在当前量化边界下，data的加权平方量化误差为WSQE："<< WSQE <<endl;
	return WSQE;
}

vector<int> Get_Quant_Bound_Relative(vector<int> A,vector<float>data,map<float,float> data_f) {
	for(int i=1; i<A.size()-1; i++) { // 枚举到每一个 Ai
		// 假定 Ai-1 和 Ai+1 已知的前提下，确定Ai的边界。
		float Summation_L = 0;
		for(auto x : data) {
			if(x >= A[i-1] && x < A[i+1]) {
				Summation_L += (1.0 / x) * data_f[x];
			}
		}
		int A_i;
		float Summation_R_Abs_Sub =  0x3f3f3f; // 记初始与  Summation_L 的最大绝对差值为一个极大值
		for(int A_i_tmp = A[i-1] + 1; A_i_tmp < A[i+1]; A_i_tmp++) {
			// 枚举全体Ai的值 且Ai 大于 Ai-1 小于 Ai+1
			// 找到与  Summation_L 最接近的 Ai 的值
			float Summation_R_Tmp = 0;
			for(auto x : data) {
				if(x >= A[i-1] && x < A_i_tmp) {
					Summation_R_Tmp += (A[i-1] / (x * x)) * data_f[x];
				}
				if(x >= A_i_tmp && x < A[i+1]) {
					Summation_R_Tmp += (A[i+1] / (x * x)) * data_f[x];
				}
			}

			if(abs(Summation_R_Tmp - Summation_L) < Summation_R_Abs_Sub) {
				Summation_R_Abs_Sub = abs(Summation_R_Tmp - Summation_L);
				A_i = A_i_tmp;
			}
		}
		A[i] = A_i; // 更新Ai
	}
	return A;
}

vector<int> Get_Quant_Bound_Absolute(vector<int> A,vector<float>data,map<float,float> data_f) {
	for(int i=1; i<A.size()-1; i++) { // 枚举到每一个 Ai
		// 假定 Ai-1 和 Ai+1 已知的前提下，确定Ai的边界。
		float Summation_L = 0;
		for(auto x : data) {
			if(x >= A[i-1] && x < A[i+1]) {
				Summation_L += (x * data_f[x]);
			}
		}
		int A_i;
		float Summation_R_Abs_Sub =  0x3f3f3f; // 记初始与  Summation_L 的最大绝对差值为一个极大值
		for(int A_i_tmp = A[i-1] + 1; A_i_tmp < A[i+1]; A_i_tmp++) {
			// 枚举全体Ai的值 且Ai 大于 Ai-1 小于 Ai+1
			// 找到与  Summation_L 最接近的 Ai 的值
			float Summation_R_Tmp = 0;
			for(auto x : data) {
				if(x >= A[i-1] && x < A_i_tmp) {
					Summation_R_Tmp += (A[i-1] * data_f[x]);
				}
				if(x >= A_i_tmp && x < A[i+1]) {
					Summation_R_Tmp += (A[i+1] * data_f[x]);
				}
			}

			if(abs(Summation_R_Tmp - Summation_L) < Summation_R_Abs_Sub) {
				Summation_R_Abs_Sub = abs(Summation_R_Tmp - Summation_L);
				A_i = A_i_tmp;
			}
		}
		A[i] = A_i; // 更新Ai
	}
	return A;
}

vector<int> Get_Quant_Bound_Weight(vector<int> A,vector<float>data,map<float,float> data_f) {

	for(int i=1; i<A.size()-1; i++) { // 枚举到每一个 Ai
		// 假定 Ai-1 和 Ai+1 已知的前提下，确定Ai的边界。
		float Summation_L = 0;
		for(auto x : data) {
			if(x >= A[i-1] && x < A[i+1]) {
				if(P == -1) {
					P = 1.0 / Cacl_Absolute_Squ_Quant(data,A,true,x);
				}
				Summation_L += (P * x * data_f[x] + (1-P) * (1.0 / x) * data_f[x]);
			}
		}
		int A_i;
		float Summation_R_Abs_Sub =  0x3f3f3f; // 记初始与  Summation_L 的最大绝对差值为一个极大值
		for(int A_i_tmp = A[i-1] + 1; A_i_tmp < A[i+1]; A_i_tmp++) {
			// 枚举全体Ai的值 且Ai 大于 Ai-1 小于 Ai+1
			// 找到与  Summation_L 最接近的 Ai 的值
			float Summation_R_Tmp = 0;
			for(auto x : data) {
				if(x >= A[i-1] && x < A_i_tmp) {
					Summation_R_Tmp += ( P * A[i-1] * data_f[x] + (1-P) * (A[i-1] / (x * x)) * data_f[x]);
				}
				if(x >= A_i_tmp && x < A[i+1]) {
					Summation_R_Tmp += ( P * A[i+1] * data_f[x] + (1-P) * (A[i+1] / (x * x)) * data_f[x]);
				}
			}

			if(abs(Summation_R_Tmp - Summation_L) < Summation_R_Abs_Sub) {
				Summation_R_Abs_Sub = abs(Summation_R_Tmp - Summation_L);
				A_i = A_i_tmp;
			}
		}
		A[i] = A_i; // 更新Ai
	}
	return A;
}

void Print_Quantization(vector<int> ve) {
	cout<<ve.size()<<endl;
	for(int i=1; i<=ve.size(); i++) {
		cout<<ve[i-1]<<" ";
		if(i % 10 == 0) cout<<endl;
	}
	cout<<endl;
}

template <typename T>
void Save_Data(char *filePath,vector<T>& data) {
	
	freopen(filePath,"w",stdout);
	for(int i=1; i<=data.size(); i++) {
		cout << data[i-1] << " ";
		if(i % 10 == 0) cout<<"\n";
	}
	
	// 关闭文件，恢复标准输出流
    fclose(stdout);
    fflush(stdout);
    cout.flush();
    freopen("CON", "w", stdout);
	
}

void Print_deatil() {
	cout<<"数据量："<<Num<<endl;
	cout<<"量化比特位："<<Bits<<endl;
	cout<<"随机数的均值："<<Mean<<endl;
	cout<<"随机数的方差："<<StdDev<<endl;
}

int main() {
	Print_deatil();
	float Min_x = 0x3f3f3f,Max_x = -0x3f3f3f;
	map<float,int> data_num; // X 出现的频次
	map<float,float> data_f; // X 出现的频率
	vector<float> data = Get_Data(Min_x,Max_x,data_num,data_f,Mean,StdDev);
	Save_Data("data.txt",data);
	cout<<"\n数据生成完毕。\n"<<endl;
	
	vector<int> A;
	A.push_back(Min_x);
	for(int i=1; i<pow(2,Bits)-1; i++) {
		A.push_back(Min_x + (i * (Max_x - Min_x)) / (pow(2,Bits) - 1) );
	}
	A.push_back(Max_x);
	Save_Data("origin_bound.txt",A);
	cout<<"\n初始量化边界以确定。\n"<<endl;
	
	//Print_Quantization(A);
	float OASQE = Cacl_Absolute_Squ_Quant(data,A); // 绝对
	float ORSQE = Cacl_Relative_Squ_Quant(data,A); // 相对
	float OWSQE = Cacl_Weight_Squ_Quant(data,A); // 加权
	
	vector<int> A_Relative = Get_Quant_Bound_Relative(A,data,data_f);
	vector<int> A_Absolute = Get_Quant_Bound_Absolute(A,data,data_f);
	vector<int> A_Weight = Get_Quant_Bound_Weight(A,data,data_f);
	Save_Data("Absolute_bound.txt",A_Absolute);
	Save_Data("Relative_bound.txt",A_Relative);
	Save_Data("Weight_bound.txt",A_Weight);
	cout<<"\n量化边界已修正。\n"<<endl;

	//Print_Quantization(A_Absolute);
	float NASQE = Cacl_Absolute_Squ_Quant(data,A_Absolute); // 绝对
	cout<<"绝对量化：相比于初始量化边界，误差减小了：" << (OASQE - NASQE) / OASQE * 100 <<"%\n" <<endl;

	//Print_Quantization(A_Relative);
	float NRSQE = Cacl_Relative_Squ_Quant(data,A_Relative); // 相对
	cout<<"相对量化：相比于初始量化边界，误差减小了：" << (ORSQE - NRSQE) / ORSQE * 100 <<"%\n" <<endl;

	float NWSQE = Cacl_Weight_Squ_Quant(data,A_Weight);   // 加权
	cout<<"加权量化：相比于初始量化边界，误差减小了：" << (OWSQE - NWSQE) / OWSQE * 100 <<"%\n" <<endl;

	return 0;
}
/**
数据量：100
量化比特位：8
随机数的均值：0
随机数的方差：100

数据生成完毕。


初始量化边界以确定。

在当前量化边界下，data的绝对平方量化误差为SQE：83.7619
在当前量化边界下，data的相对平方量化误差为RSQE：195.155
在当前量化边界下，data的加权平方量化误差为WSQE：0.987237

量化边界已修正。

在当前量化边界下，data的绝对平方量化误差为SQE：106.665
绝对量化：相比于初始量化边界，误差减小了：-27.3431%

在当前量化边界下，data的相对平方量化误差为RSQE：203.006
相对量化：相比于初始量化边界，误差减小了：-4.02271%

在当前量化边界下，data的加权平方量化误差为WSQE：0.16467
加权量化：相比于初始量化边界，误差减小了：83.3201%

----------------------------------------------------------------
数据量：100
量化比特位：8
随机数的均值：0
随机数的方差：500

数据生成完毕。


初始量化边界以确定。

在当前量化边界下，data的绝对平方量化误差为SQE：2002.78
在当前量化边界下，data的相对平方量化误差为RSQE：199.037
在当前量化边界下，data的加权平方量化误差为WSQE：3.17848

量化边界已修正。

在当前量化边界下，data的绝对平方量化误差为SQE：1167.84
绝对量化：相比于初始量化边界，误差减小了：41.6892%

在当前量化边界下，data的相对平方量化误差为RSQE：211.767
相对量化：相比于初始量化边界，误差减小了：-6.39563%

在当前量化边界下，data的加权平方量化误差为WSQE：2.05575
加权量化：相比于初始量化边界，误差减小了：35.323%

----------------------------------------------------------------
数据量：100
量化比特位：8
随机数的均值：0
随机数的方差：1000

数据生成完毕。


初始量化边界以确定。

在当前量化边界下，data的绝对平方量化误差为SQE：7990.04
在当前量化边界下，data的相对平方量化误差为RSQE：190.387
在当前量化边界下，data的加权平方量化误差为WSQE：3.46398

量化边界已修正。

在当前量化边界下，data的绝对平方量化误差为SQE：3943.21
绝对量化：相比于初始量化边界，误差减小了：50.6484%

在当前量化边界下，data的相对平方量化误差为RSQE：211.554
相对量化：相比于初始量化边界，误差减小了：-11.118%

在当前量化边界下，data的加权平方量化误差为WSQE：2.15767
加权量化：相比于初始量化边界，误差减小了：37.711%

----------------------------------------------------------------
数据量：100
量化比特位：8
随机数的均值：432
随机数的方差：43232

数据生成完毕。


初始量化边界以确定。

在当前量化边界下，data的绝对平方量化误差为SQE：1.50339e+011
在当前量化边界下，data的相对平方量化误差为RSQE：173.704
在当前量化边界下，data的加权平方量化误差为WSQE：3.64337

量化边界已修正。

在当前量化边界下，data的绝对平方量化误差为SQE：1.46033e+011
绝对量化：相比于初始量化边界，误差减小了：2.86425%

在当前量化边界下，data的相对平方量化误差为RSQE：166.046
相对量化：相比于初始量化边界，误差减小了：4.40882%

在当前量化边界下，data的加权平方量化误差为WSQE：1.92863
加权量化：相比于初始量化边界，误差减小了：47.0647%

----------------------------------------------------------------
数据量：100
量化比特位：8
随机数的均值：0
随机数的方差：43232

数据生成完毕。


初始量化边界以确定。

在当前量化边界下，data的绝对平方量化误差为SQE：1.50339e+011
在当前量化边界下，data的相对平方量化误差为RSQE：161.456
在当前量化边界下，data的加权平方量化误差为WSQE：3.64258

量化边界已修正。

在当前量化边界下，data的绝对平方量化误差为SQE：1.41738e+011
绝对量化：相比于初始量化边界，误差减小了：5.72112%

在当前量化边界下，data的相对平方量化误差为RSQE：185.786
相对量化：相比于初始量化边界，误差减小了：-15.0693%

在当前量化边界下，data的加权平方量化误差为WSQE：1.92867
加权量化：相比于初始量化边界，误差减小了：47.0522%

**/
