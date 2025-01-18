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
	//cache��С
	int cacheSize;
	//cache����������ھ����ϵ�λ�ã���˼��cache�洢�Ӿ����rowStart�С�colStart�е�rowEnd�С�colEnd��
	int rowStart;//��ʾ��ʼ������
	int colStart;//��ʾ��ʼ������
	int rowEnd;//��ʾ����������
	int colEnd;//��ʾ����������
	vector<int> myCache;
	bool flag;//�����Ƿ񱻸Ķ�
	bool isW;//�����Ƿ�������
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

//д���棬�������ļ����дӵ�rS�С�cS�е�����д�뻺���У�������д����
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
			//��ǰ���Ѷ��꣬����һ��
			++rS;
			if (rS == (matrix.getRowNum()))
			{
				//�������Ѷ��꣬д�������
				break;
			}
			//ÿ�дӵ�һ�п�ʼ��
			cS = 0;
		}
		//д��cache
		myCache[now++] = myMatrix[rS][cS++];
	}
	rowEnd = rS;
	colEnd = cS - 1;
}

//д�أ�����������д�ؾ����ļ���
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
			//��ǰ����д�꣬д��һ��
			++rS;
			//ÿ�дӵ�һ�п�ʼ��
			cS = 0;
		}
	}
}

//�����ھ����е�λ�ö�����
//���������û��Ҫ���������һ����������ѱ��Ķ����򽫻�������д�ؾ����ļ���
int Cache::readCache(int r, int c, Matrix matrix)
{
	int count0 = matrix.getColNum() * r + c;
	int count1 = rowStart * matrix.getColNum() + colStart;
	int count = count0 - count1;
	//cout << count << " ";
	if (!isInCache(r, c))
	{
		//���ڻ�����
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