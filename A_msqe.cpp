#include <bits/stdc++.h>
using namespace std;

int Num = 0;
const bool LH = true;
const int Bits = 5;
const float Mean = 500.0;
const float StdDev = 43232.0f;
float P = -1; // 此处若设置 P 为 -1 则按 1/error 加权

int Pow(int x,int y){
	int ans = 1;
	for(int i=0;i<y;i++){
		ans *= x;
	}
	return ans;
}


int main() {
	vector<float> data,weight,original_data;
	freopen("11.txt","r",stdin);
	float Min = 0x3f3f,Max = -0x3f3f;
	float x,kk=0;
	while(cin>>x){
		data.push_back(x);
		Min = min(Min,x);
		Max = max(Max,x);
		kk++;
	}
	cout<<kk<<" "<<Min<<" "<<Max<<endl; 
	// 关闭文件，恢复标准输出流
    fclose(stdin);
    fflush(stdin);
    freopen("CON", "r", stdin);
	//cout<<Min<<" "<<Max<<endl;
	int n_clusters = Pow(2,Bits) - 1;
	weight = original_data = data;
	vector<float> bound,bound_msqe,original_bound;
	bound.push_back(Min);
	for(int i=1;i<n_clusters;i++){
		bound.push_back(Min + i*(Max-Min)/n_clusters);
	}
	bound.push_back(Max);
	
	bound_msqe = bound;
	for(auto x : bound_msqe){
		cout<<x<<' ';
	}
	cout<<endl;
	cout<<"*******************************************\n" ;
	
	if(LH){
		vector<float> input_vector = weight;
		sort(input_vector.begin(),input_vector.end());
		for(int c=0;c<1000000;c++){
			vector<float> msqe_tmp = bound_msqe;
			
			for(int i=1;i<n_clusters;i++){
				
				vector<int> index;
				vector<float> tmp;
				float sum = 0; 
				for(int j=0;j<weight.size();j++){
					if(input_vector[j] >= bound_msqe[i+1]){
						break;
					}
					if(input_vector[j] >= bound_msqe[i-1] && input_vector[j] < bound_msqe[i+1]){ // find x_c 
						index.push_back(j);
						tmp.push_back(input_vector[j]);
						sum += input_vector[j];
					}
				}
				if(index.size() == 0){
					bound_msqe[i] = bound[i];
				}else{
					float temp = (bound_msqe[i + 1] * index.size() - sum) / (bound_msqe[i + 1] - bound_msqe[i - 1]);
					//cout<<bound_msqe[i] << " " <<bound_msqe[i + 1]<<endl;
					//cout<<int(temp)<<" "<<index[0]<<endl;;
					bound_msqe[i] = input_vector[int(temp)+index[0]];
					if(bound_msqe[i] >= bound_msqe[i + 1])
						bound_msqe[i] = bound[i];
				}
			} 
			
			if(msqe_tmp == bound_msqe) break;
		}
	}
	
	for(auto x : bound_msqe){
		cout<<x<<' ';
	}
	cout<<endl;
	
	
	
	for(int i=1;i<=n_clusters;i++){
		float k_Max = bound_msqe[i];
		float k_Min = bound_msqe[i-1];
		vector<int> index;
		vector<float> tmp,p_Min;
		for(int j=0;j<weight.size();j++){
			if(weight[j] >= bound_msqe[i-1] && weight[j] < bound_msqe[i]){
				index.push_back(j);
				tmp.push_back(weight[j]);
				if(k_Max != k_Min){
					p_Min.push_back((k_Max - weight[j]) / (k_Max - k_Min));
				}else{
					p_Min.push_back(1.0f);
				}
			}
		}
		
		// 使用随机数生成器
	    std::random_device rd;
	    std::mt19937 gen(rd());
	    std::uniform_real_distribution<float> dis(0.0, 1.0);
	    // 生成服从均匀分布的随机数
	    for (int j = 0; j < tmp.size(); ++j) {
	        float random_value = dis(gen);  // 生成 [0, 1) 范围内的随机数
	        float prob = random_value * p_Min[j];
	
	        // 随机量化
	        if (prob < p_Min[i]) {
	            tmp[j] = k_Min;
	        } else {
	            tmp[j] = k_Max;
	        }
	    }
		for(int j=0;j<index.size();j++){
			weight[index[j]] = tmp[j];
		}
	}
	
	float q = 0;
	for(int i=0;i<data.size();i++){
		q = q + (Pow((original_data[i] - weight[i]),2));
	}
	freopen("out.txt","w",stdout);
	int k = 1;
	for(int i=0;i<data.size();i++){
		cout<< weight[i] << " ";
		if((i+1) % 5 == 0){
			cout<<endl;
			k++;
			if(k % 5 == 0){
				cout<<endl;
			}
		} 
		
	}
	cout<<q<<endl;
	return 0;
}

