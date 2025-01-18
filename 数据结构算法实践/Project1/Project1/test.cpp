#include <iostream>
#include <vector>
#include <string>
#include "Cache.h"

using namespace std;

const int cacheSize = 5;
const int n = 5;

int multiplyByIJK();
int multiplyByIKJ();

int main()
{
	cout << multiplyByIJK() << endl;
}

int multiplyByIJK()
{
	Matrix A(n, n);
	Matrix B(n, n);
	Matrix C(n, n);
	Cache aC(cacheSize);
	Cache bC(cacheSize);
	Cache cC(cacheSize);
	cout << "矩阵A的内容：" << endl;
	A.showMyMatrix();
	aC.writeCache(0, 0, A);
	aC.outAll();
	cout << "缓存A的内容：" << endl;
	aC.showCache();

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				int a = aC.readCache(i, k, A);
				int b = bC.readCache(k, j, B);
				int c = cC.readCache(i, j, C);
				c += a * b;
				cC.wCache(i, j, n, c);
			}
		}
	}
	cout << "a b c的缺失次数：" << endl;
	cout << aC.getMissNum() << endl;
	cout << bC.getMissNum() << endl;
	cout << cC.getMissNum() << endl;
	return aC.getMissNum() + bC.getMissNum() + cC.getMissNum();
}
