#include <bits/stdc++.h>
using namespace std;

const int Num = 100; 
const int Bits = 8;
float P = -1; // �˴������� P Ϊ -1 �� 1/error ��Ȩ 

vector<float> Get_Data(float &Min_x,float &Max_x,map<float,int> &data_num, map<float,float> &data_f,float mean = 0.0f, float stddev = 1.0f) {
    // ����������������̬�ֲ���
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, stddev);
    // �洢���ɵ������
    std::vector<float> data;
    // ����ָ�������������
    for (int i = 0; i < Num; ++i) {
        float value = distribution(generator);
        data.push_back(value);
        data_num[value] ++;
        Min_x = min(Min_x,value);
        Max_x = max(Max_x,value);
    }
    for(auto x : data_num){
    	float f = x.second * 1.0 / Num;
    	data_f[x.first] = f;
	}

    return data;
}

float Cacl_Relative_Squ_Quant(vector<float> data,vector<int> A){
	float RSQE = 0; // ���������� 
	for(auto x : data){
		int A_R_index = lower_bound(A.begin(),A.end(),x) - A.begin(); // �ҵ���һ�����ڵ��� x �ı߽�  ��Ϊ�ұ߽�Ai�� 
		if(A_R_index == 0) A_R_index = 1; // ��ֹԽ�� 
		if(A_R_index == A.size()) A_R_index = A.size()-1; // ��ֹԽ�� 
		float e = ( -1 + (1.0*(A[A_R_index] + A[A_R_index-1])) / x + ((1.0)*(A[A_R_index-1] * A[A_R_index])) / (x * x) );
		RSQE += e;
	}
	cout<<"\n�ڵ�ǰ�����߽��£�data�����ƽ���������ΪRSQE��"<< RSQE <<endl<<endl;
	return RSQE;
}

float Cacl_Absolute_Squ_Quant(vector<float> data, vector<int> A, bool point = false, float p_x = 0.0){
	float SQE = 0; // ����������� 
	if(point == false){
		for(auto x : data){
			int A_R_index = lower_bound(A.begin(),A.end(),x) - A.begin(); // �ҵ���һ�����ڵ��� x �ı߽�  ��Ϊ�ұ߽�Ai�� 
			if(A_R_index == 0) A_R_index = 1; // ��ֹԽ�� 
			if(A_R_index == A.size()) A_R_index = A.size()-1; // ��ֹԽ�� 
			float e = -1 * (x * x) + (A[A_R_index - 1] + A[A_R_index]) * x - (A[A_R_index-1] * A[A_R_index]) ;
			SQE += e;
		}
		cout<<"\n�ڵ�ǰ�����߽��£�data�ľ���ƽ���������ΪSQE��"<< SQE <<endl<<endl;
		return SQE;
	}else{
		int A_R_index = lower_bound(A.begin(),A.end(),p_x) - A.begin(); // �ҵ���һ�����ڵ��� x �ı߽�  ��Ϊ�ұ߽�Ai�� 
		if(A_R_index == 0) A_R_index = 1; // ��ֹԽ�� 
		if(A_R_index == A.size()) A_R_index = A.size()-1; // ��ֹԽ�� 
		float e = -1 * (p_x * p_x) + (A[A_R_index - 1] + A[A_R_index]) * p_x - (A[A_R_index-1] * A[A_R_index]) ;
		return e;
	}
	
}

float Cacl_Weight_Squ_Quant(vector<float> data,vector<int> A){
	float WSQE = 0; // Ȩ��������� 
	for(auto x : data){
		int A_R_index = lower_bound(A.begin(),A.end(),x) - A.begin(); // �ҵ���һ�����ڵ��� x �ı߽�  ��Ϊ�ұ߽�Ai�� 
		if(A_R_index == 0) A_R_index = 1; // ��ֹԽ�� 
		if(A_R_index == A.size()) A_R_index = A.size()-1; // ��ֹԽ�� 
		float Relative_Error = ( -1 + (1.0*(A[A_R_index] + A[A_R_index-1])) / x + ((1.0)*(A[A_R_index-1] * A[A_R_index])) / (x * x) );
		float Absolute_Error = -1 * (x * x) + (A[A_R_index - 1] + A[A_R_index]) * x - (A[A_R_index-1] * A[A_R_index]) ;
		if(P == -1) P = 1.0 / Absolute_Error;
		WSQE = P*Absolute_Error + (1-P) * Relative_Error;
	}
	cout<<"\n�ڵ�ǰ�����߽��£�data�ļ�Ȩƽ���������ΪWSQE��"<< WSQE <<endl<<endl;
	return WSQE;
}

vector<int> Get_Quant_Bound_Relative(vector<int> A,vector<float>data,map<float,float> data_f){
	for(int i=1;i<A.size()-1;i++){ // ö�ٵ�ÿһ�� Ai 
		// �ٶ� Ai-1 �� Ai+1 ��֪��ǰ���£�ȷ��Ai�ı߽硣
		float Summation_L = 0;
		for(auto x : data){
			if(x >= A[i-1] && x < A[i+1]){
				Summation_L += (1.0 / x) * data_f[x];
			}
		}
		int A_i;
		float Summation_R_Abs_Sub =  0x3f3f3f; // �ǳ�ʼ��  Summation_L �������Բ�ֵΪһ������ֵ 
		for(int A_i_tmp = A[i-1] + 1; A_i_tmp < A[i+1]; A_i_tmp++){
		 // ö��ȫ��Ai��ֵ ��Ai ���� Ai-1 С�� Ai+1 
		 // �ҵ���  Summation_L ��ӽ��� Ai ��ֵ
			float Summation_R_Tmp = 0;
			for(auto x : data){
				if(x >= A[i-1] && x < A_i_tmp){
					Summation_R_Tmp += (A[i-1] / (x * x)) * data_f[x];
				}
				if(x >= A_i_tmp && x < A[i+1]){
					Summation_R_Tmp += (A[i+1] / (x * x)) * data_f[x];
				}
			}
			
			if(abs(Summation_R_Tmp - Summation_L) < Summation_R_Abs_Sub){
				Summation_R_Abs_Sub = abs(Summation_R_Tmp - Summation_L);
				A_i = A_i_tmp;
			}
		}
		A[i] = A_i; // ����Ai 
	}
	return A;
}
vector<int> Get_Quant_Bound_Absolute(vector<int> A,vector<float>data,map<float,float> data_f){
	for(int i=1;i<A.size()-1;i++){ // ö�ٵ�ÿһ�� Ai 
		// �ٶ� Ai-1 �� Ai+1 ��֪��ǰ���£�ȷ��Ai�ı߽硣
		float Summation_L = 0;
		for(auto x : data){
			if(x >= A[i-1] && x < A[i+1]){
				Summation_L += (x * data_f[x]);
			}
		}
		int A_i;
		float Summation_R_Abs_Sub =  0x3f3f3f; // �ǳ�ʼ��  Summation_L �������Բ�ֵΪһ������ֵ 
		for(int A_i_tmp = A[i-1] + 1; A_i_tmp < A[i+1]; A_i_tmp++){
		 // ö��ȫ��Ai��ֵ ��Ai ���� Ai-1 С�� Ai+1 
		 // �ҵ���  Summation_L ��ӽ��� Ai ��ֵ
			float Summation_R_Tmp = 0;
			for(auto x : data){
				if(x >= A[i-1] && x < A_i_tmp){
					Summation_R_Tmp += (A[i-1] * data_f[x]);
				}
				if(x >= A_i_tmp && x < A[i+1]){
					Summation_R_Tmp += (A[i+1] * data_f[x]);
				}
			}
			
			if(abs(Summation_R_Tmp - Summation_L) < Summation_R_Abs_Sub){
				Summation_R_Abs_Sub = abs(Summation_R_Tmp - Summation_L);
				A_i = A_i_tmp;
			}
		}
		A[i] = A_i; // ����Ai 
	}
	return A;
}

vector<int> Get_Quant_Bound_Weight(vector<int> A,vector<float>data,map<float,float> data_f){
	
	for(int i=1;i<A.size()-1;i++){ // ö�ٵ�ÿһ�� Ai 
		// �ٶ� Ai-1 �� Ai+1 ��֪��ǰ���£�ȷ��Ai�ı߽硣
		float Summation_L = 0;
		for(auto x : data){
			if(x >= A[i-1] && x < A[i+1]){
				if(P == -1){
					P = 1.0 / Cacl_Absolute_Squ_Quant(data,A,true,x);
				}
				Summation_L += (P * x * data_f[x] + (1-P) * (1.0 / x) * data_f[x]);
			}
		}
		int A_i;
		float Summation_R_Abs_Sub =  0x3f3f3f; // �ǳ�ʼ��  Summation_L �������Բ�ֵΪһ������ֵ 
		for(int A_i_tmp = A[i-1] + 1; A_i_tmp < A[i+1]; A_i_tmp++){
		 // ö��ȫ��Ai��ֵ ��Ai ���� Ai-1 С�� Ai+1 
		 // �ҵ���  Summation_L ��ӽ��� Ai ��ֵ
			float Summation_R_Tmp = 0;
			for(auto x : data){
				if(x >= A[i-1] && x < A_i_tmp){
					Summation_R_Tmp += ( P * A[i-1] * data_f[x] + (1-P) * (A[i-1] / (x * x)) * data_f[x]);
				}
				if(x >= A_i_tmp && x < A[i+1]){
					Summation_R_Tmp += ( P * A[i+1] * data_f[x] + (1-P) * (A[i+1] / (x * x)) * data_f[x]);
				}
			}
			
			if(abs(Summation_R_Tmp - Summation_L) < Summation_R_Abs_Sub){
				Summation_R_Abs_Sub = abs(Summation_R_Tmp - Summation_L);
				A_i = A_i_tmp;
			}
		}
		A[i] = A_i; // ����Ai 
	}
	return A;
}

void Print_Quantization(vector<int> ve){
	cout<<ve.size()<<endl;
	for(int i=1;i<=ve.size();i++){
		cout<<ve[i-1]<<" ";
		if(i % 10 == 0) cout<<endl;
	}
	cout<<endl; 
}

int main(){
	
	float Min_x = 0x3f3f3f,Max_x = -0x3f3f3f;                 
	map<float,int> data_num; // X ���ֵ�Ƶ��
	map<float,float> data_f; // X ���ֵ�Ƶ�� 
	vector<float> data = Get_Data(Min_x,Max_x,data_num,data_f,324,43232.0);
	
	vector<int> A;A.push_back(Min_x);
	for(int i=1;i<pow(2,Bits)-1;i++){
		A.push_back(Min_x + (i * (Max_x - Min_x)) / (pow(2,Bits) - 1) );
	}
	A.push_back(Max_x);
	
	Print_Quantization(A);
	float OASQE = Cacl_Absolute_Squ_Quant(data,A); // ���� 
	float ORSQE = Cacl_Relative_Squ_Quant(data,A); // ��� 
	float OWSQE = Cacl_Weight_Squ_Quant(data,A); // ��Ȩ 
	
	vector<int> A_Relative = Get_Quant_Bound_Relative(A,data,data_f);
	vector<int> A_Absolute = Get_Quant_Bound_Absolute(A,data,data_f);
	vector<int> A_Weight = Get_Quant_Bound_Weight(A,data,data_f);
	
	Print_Quantization(A_Absolute);
	float NASQE = Cacl_Absolute_Squ_Quant(data,A_Absolute); // ���� 
	cout<<"\n���������֮ǰ������С�ˣ�" << (OASQE - NASQE) / OASQE * 100 <<"%" <<endl;
	
	Print_Quantization(A_Relative);
	float NRSQE = Cacl_Relative_Squ_Quant(data,A_Relative); // ��� 
	cout<<"\n���������֮ǰ������С�ˣ�" << (ORSQE - NRSQE) / ORSQE * 100 <<"%" <<endl;
	
	float NWSQE = Cacl_Weight_Squ_Quant(data,A_Weight);   // ��Ȩ 
	cout<<"\n���������֮ǰ������С�ˣ�" << (OWSQE - NWSQE) / OWSQE * 100 <<"%" <<endl;
	
	return 0;
}
