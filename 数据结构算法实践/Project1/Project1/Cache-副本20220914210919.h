//Cache.h
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "Matrix.h"

using namespace std;

class Cache {
public:
	Cache(int cSize);
	void writeCache(int rS, int cS, Matrix matrix);
	void wCache(int r, int c, int rSize, int number);
	void writeBack(Matrix matrix);
	int readCache(int r, int c, Matrix matrix);
	bool isInCache(int r, int c);
	void showCache();
	int getMissNum();
	void outAll();

private:
	//cache大小
	int cacheSize;
	//cache储存的数据在矩阵上的位置，意思是cache存储从矩阵的rowStart行、colStart列到rowEnd行、colEnd列
	int rowStart;//表示开始的行数
	int colStart;//表示开始的列数
	int rowEnd;//表示结束的行数
	int colEnd;//表示结束的行数
	vector<int> myCache;
	bool flag;//缓存是否被改动
	bool isW;//缓存是否有内容
	int missNum;
};

Cache::Cache(int cSize)
{
	cacheSize = cSize;
	myCache.resize(cacheSize);
	flag = false;
	rowStart = 0;
	colStart = 0;
	rowEnd = 0;
	colEnd = 0;
	missNum = 0;
	isW = false;
}

//写缓存，将矩阵（文件）中从第rS行、cS列的数据写入缓存中，将缓存写满。
void Cache::writeCache(int rS, int cS, Matrix matrix)
{
	isW = true;
	vector<vector<int>> myMatrix = matrix.getMyMatrix();
	rowStart = rS;
	colStart = cS;
	rowEnd = rS;
	colEnd = cS;
	int now = 0;
	while (now < cacheSize)
	{
		if (cS == (matrix.getColNum()))
		{
			//当前行已读完，读下一行
			++rS;
			if (rS == (matrix.getRowNum()))
			{
				//所有行已读完，写缓存完成
				break;
			}
			//每行从第一列开始读
			cS = 0;
		}
		//写入cache
		myCache[now++] = myMatrix[rS][cS++];
	}
	rowEnd = rS;
	colEnd = cS - 1;
}

//写回，将缓存内容写回矩阵（文件）
void Cache::writeBack(Matrix matrix)
{
	int rS = rowStart;
	int cS = colStart;
	int now = 0;
	vector<vector<int>> myM = matrix.getMyMatrix();

	while (rS < rowEnd && cS < colEnd)
	{
		myM[rS][cS++] = myCache[now++];
		if (cS == (matrix.getColNum()))
		{
			//当前行已写完，写下一行
			++rS;
			//每行从第一列开始读
			cS = 0;
		}
	}
}

//根据在矩阵中的位置读缓存
//如果缓存中没有要读的数据且缓存中数据已被改动，则将缓存数据写回矩阵（文件）
int Cache::readCache(int r, int c, Matrix matrix)
{
	int count0 = matrix.getColNum() * r + c;
	int count1 = rowStart * matrix.getColNum() + colStart;
	int count = count0 - count1;
	//cout << count << " ";
	if (!isInCache(r, c))
	{
		//不在缓存中
		missNum++;
		if (flag)
		{
			writeBack(matrix);
		}
		writeCache(r, c, matrix);
		count = 0;
	}
	//cout << count << endl;
	return myCache[count];
}

bool Cache::isInCache(int r, int c)
{
	if (!isW)
	{
		return false;
	}
	if (r >= rowStart && r <= rowEnd
		&& c >= colStart && c <= colEnd)
	{
		return true;
	}
	return false;
}

void Cache::showCache()
{
	for (int i = 0; i < myCache.size(); i++)
	{
		cout << myCache[i] << " ";
	}
	cout << endl;
}

int Cache::getMissNum()
{
	return missNum;
}

void Cache::outAll()
{
	cout << "rowStart " << rowStart << endl;
	cout << "colStart " << colStart << endl;
	cout << "rowEnd " << rowEnd << endl;
	cout << "colEnd " << colEnd << endl;
}

void Cache::wCache(int r, int c, int rSize, int number)
{
	int count0 = r * rSize + c;
	int count1 = rowStart * rSize + colStart;
	int count = count0 - count1;
	myCache[count] = number;
	flag = true;
}