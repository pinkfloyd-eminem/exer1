#include <iostream>
#include <cstdlib> 
#include <ctime>
#include <windows.h>
#include<time.h>

using namespace std;

void Random(int* a, int n, int l, int r)//填充范围在l~r的随机数数组
{
	srand(time(0));  //设置时间种子
	for (int i = 0; i < n; i++) 
	{
		a[i] = rand() % (r - l + 1) + l;//生成区间r~l的随机数 
	}
}
int Random(int l, int r)           //生成范围在l和r间的随机数
{
	int ra;
	srand((unsigned)time(NULL));
	ra = rand() % (r - l + 1) + l;
	return ra;
}
void Swap(int a[], int i, int r)      //交换数组元素
{
	int temp;
	temp = a[i];
	a[i] = a[r];
	a[r] = temp;
}
int Partition(int a[], int p, int r) 
{
	int x = a[r];
	int i = p - 1;
	for (int j = p; j <= r - 1; j++)
	{
		if (a[j] <= x)
		{
			i = i + 1;
			Swap(a, i, j);
		}
	}
	Swap(a, i + 1, r);
	return i + 1;
}
int Random_Partition(int a[], int p, int r)       
{
	int i = Random(p, r);
	Swap(a, i, r);
	return Partition(a, p, r);
}

void Random_QuickSort(int a[], int p, int r)      //随机化版本快速排序
{
	int q;
	if (p < r)
	{
		q = Random_Partition(a, p, r);
		Random_QuickSort(a, p, q - 1);
		Random_QuickSort(a, q + 1, r);
	}
}
void QuickSort(int a[], int p, int r)              //普通快速排序
{ 
	int q;
	if (p < r)
	{
		q = Partition(a, p, r);
		QuickSort(a, p, q - 1);
		QuickSort(a, q + 1, r);
	}
}
int main()
{
	int n = 1000000;//数组元素的个数，即生成随机数的个数
	DWORD start, end;
	int* a = new int[n];
	//int* b = new int[n];
	Random(a, n, 1, 30000);
	QuickSort(a, 0, n);
	start = timeGetTime();
	//QuickSort(a, 0, n - 1);
	//Random(a, n, 1, 30000);//生成随机数的通常范围为0~32767，这里通过取模控制取值为0~100 
	//Random(b, n, 1, 30000);
	Random_QuickSort(a, 0, n-1);
	//QuickSort(a, 0, n-1);
	end = timeGetTime() - start;
    #pragma comment(lib, "winmm.lib")
	cout << "运行时间:" << end << endl;
	return 0;
}

