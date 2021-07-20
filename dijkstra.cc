#include<iostream>
#include<iomanip>
#include<cstring>
#include<stack>

#define MAX 100
#define INF 0x3f3f3f3f
int dis[MAX], path[MAX],vis[MAX];
using namespace std;
struct MGraph
{
	int EList[MAX][MAX];
	int n, e;
}G;
void init()
{
	memset(G.EList, INF, sizeof(G.EList));  //所有边权重重置为INF
}
void addEdge(int u,int v,int weight)
{
	G.EList[u][v] = weight;
}
void dijkstra(MGraph G, int s, int dis[], int path[])
{
	int min, i, j, v;
	/*
	*初始化，松弛源点的邻节点
	*/
	for (int i = 0; i < G.n; i++)
	{
		dis[i] = G.EList[s][i];
		vis[i] = 0;
		if (G.EList[s][i] < INF)
		{
			path[i] = s;
		}
		else
		{
			path[i] = -1;
		}
	}
	vis[s] = 1;
	path[s] = -1;
	/**********迭代***********/
	for (i = 0; i < (G.n-1); i++)
	{
		min = INF;
		for (j = 0; j < G.n; j++)
		{
			if (vis[j] == 0 && dis[j] < min)
			{
				v = j;
				min = dis[j];
			}
		}
		vis[v] = 1;
		for (j = 0; j < G.n; j++)
		{
			if (vis[j] == 0 && dis[v] + G.EList[v][j] < dis[j])
			{
				dis[j] = dis[v] + G.EList[v][j];
				path[j] = v;
			}
		}
	}
}
void showPath(int path[], int t)
/****path[]为路径记录数组，参数t为终点****/
{
	stack<int> stk;
	while (path[t] != -1)
	{
		stk.push(t);
		t = path[t];
	}
	stk.push(t);
	cout << "最短路径：" << endl;
	while (!stk.empty())
	{
		cout << stk.top() << " ";
		stk.pop();
	}
	cout << endl;
}
int main()
{
	int m, n;   //边数，节点数
	int x, y, w;  //节点x,节点y，边权重x->y
	int s,t;      //源点，终点
	init();
	cout << "请输入边数 节点数：" << endl;
	cin >> m >> n;
	G.e = m;
	G.n = n;
	cout << "请输入邻接矩阵：" << endl;
	for (int i = 0; i < m; i++)
	{
		cin >> x >> y >> w;
		addEdge(x, y, w);
	}
	cout << "请输入源点：" << endl;
	cin >> s;
	dijkstra(G, s, dis, path);
	cout << "请输入终点：" << endl;
	cin >> t;
	showPath(path, t);
}

