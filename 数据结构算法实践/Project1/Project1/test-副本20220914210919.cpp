//test.cpp
#include <iostream>
#include <vector>
#include <string>
#include "Cache.h"

using namespace std;

int multiplyByIJK(int cacheSize, int n);
int multiplyByIKJ(int cacheSize, int n);

int main()
{
	multiplyByIJK(5, 5);
	multiplyByIJK(5, 10);
	multiplyByIJK(10, 10);
	multiplyByIJK(10, 100);
	multiplyByIJK(100, 100);

	multiplyByIKJ(5, 5);
	multiplyByIKJ(5, 10);
	multiplyByIKJ(10, 10);
	multiplyByIKJ(10, 100);
	multiplyByIKJ(100, 100);
}

int multiplyByIJK(int cacheSize, int n)
{
	cout << "IJK" << endl;
	cout << "缓存大小 为 " << cacheSize << "，矩阵规模为" << n << " * " << n << endl;
	Matrix A(n, n);
	Matrix B(n, n);
	Matrix C(n, n);
	Cache aC(cacheSize);
	Cache bC(cacheSize);
	Cache cC(cacheSize);

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
	cout << "a b c的缺失次数分别为：" << endl;
	cout << aC.getMissNum() <<" ";
	cout << bC.getMissNum() << " ";
	cout << cC.getMissNum() << endl;
	cout << "总的缺失次数为：";
	cout<< aC.getMissNum() + bC.getMissNum() + cC.getMissNum()<<endl << endl;
	return aC.getMissNum() + bC.getMissNum() + cC.getMissNum();
}

int multiplyByIKJ(int cacheSize, int n)
{
	cout << "IKJ" << endl;
	cout << "缓存大小 为 " << cacheSize << "，矩阵规模为" << n << " * " << n << endl;
	Matrix A(n, n);
	Matrix B(n, n);
	Matrix C(n, n);
	Cache aC(cacheSize);
	Cache bC(cacheSize);
	Cache cC(cacheSize);

	for (int i = 0; i < n; i++)
	{
		for (int k = 0; k < n; k++)
		{
			for (int j = 0; j < n; j++)
			{
				int a = aC.readCache(i, k, A);
				int b = bC.readCache(k, j, B);
				int c = cC.readCache(i, j, C);
				c += a * b;
				cC.wCache(i, j, n, c);
			}
		}
	}
	cout << "a b c的缺失次数分别为：" << endl;
	cout << aC.getMissNum() << " ";
	cout << bC.getMissNum() << " ";
	cout << cC.getMissNum() << endl;
	cout << "总的缺失次数为：";
	cout << aC.getMissNum() + bC.getMissNum() + cC.getMissNum()<<endl << endl;
	return aC.getMissNum() + bC.getMissNum() + cC.getMissNum();
}
