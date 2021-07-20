#include<iostream>
#include<string>
#include<stdlib.h>
#include<stack>
#define MAX_NUM 100
using namespace std;
void longest_common_subsequence(int dp[MAX_NUM][MAX_NUM],string s1,string s2);
void init(int dp[MAX_NUM][MAX_NUM]);      //dp数组，邻接矩阵形式
void show_subsequence(int dp[MAX_NUM][MAX_NUM],string s1,string s2);
int main()
{
	int dp[MAX_NUM][MAX_NUM];     //dp数组;
	string s1 = "adfrghyj";
	string s2 = "afgyj";
	init(dp);
	longest_common_subsequence(dp, s1, s2);
	cout << "LCS长度："<<dp[s1.length()][s2.length()]<<endl;
	cout << "LCS：" << endl;
	show_subsequence(dp, s1, s2);
}
void longest_common_subsequence(int dp[MAX_NUM][MAX_NUM], string s1, string s2)
{
	int m = s1.length();
	int n = s2.length();
	for (int i = 1; i <= m; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			if (s1[i - 1] == s2[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			else
			{
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
	}
}
void init(int dp[MAX_NUM][MAX_NUM])
{
	for (int i = 0; i < MAX_NUM; i++)
	{
		dp[0][i] = 0;
		dp[i][0] = 0;
	}
}
void show_subsequence(int dp[MAX_NUM][MAX_NUM], string s1, string s2)  //打印一个最长公共子序列
{
	int m = s1.length()-1;
	int n = s2.length()-1;
	stack<char> stk;        //栈结构
	while (m >= 0 && n >= 0)
	{
		if (s1[m] == s2[n])
		{
			stk.push(s1[m]);
			m--;
			n--;
		}
		else
		{
			if (dp[m][n+1] > dp[m+1][n]) 
			{
				m--;
			}
			else
			{
				n--;	
			}
		}
	}
	cout << "最长公共子序列：" << endl;
	while (!stk.empty())
	{
		cout << stk.top() << " ";
		stk.pop();
	}
	cout << endl;
}

