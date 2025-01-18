//MergeSort.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include<algorithm>
#include "CBuffer.h"

using namespace std;

const int bufferSize = 10;

//生成要排序的磁盘中的大文件，文件用数组来替代
vector<int> initData(int allLength);

//Run generation 生成排序好的串 返回的vector<int> 为串的大小信息
vector<int> runGeneration(vector<int>* a, int rSize);

//输出所有的串
void showRuns(vector<int>* a, vector<int>* runInfo);

//对所有的串进行归并排序
void mergeSort(vector<int>* a, vector<int>* runInfo);
void merge(vector<int>* a, vector<int>* runInfo, int run0, int run1);

//模拟归并排序实现外部排序。输入参数为磁盘中大文件总的长度和生成的每个串的大小
void externalMergeSort(int length, int runSize);

int main()
{
	/*externalMergeSort(100, 10);
	externalMergeSort(100, 20);
	externalMergeSort(1000, 50);
	externalMergeSort(10000, 100);*/
	externalMergeSort(100, 10);
	

}

vector<int> initData(int allLength)
{
	vector<int> a;
	a.resize(allLength);
	srand(time(0));
	for (int i = 0; i < allLength; i++)
	{
		a[i] = rand() % 100 + 1;
	}
	return a;
}


vector<int> runGeneration(vector<int>* a, int rSize)
{
	vector<int> runInfo;
	int runNumber = (a->size() - 1) / rSize + 1;
	runInfo.resize(runNumber);

	int i = 0;
	for (; i < runNumber - 1; i++)
	{
		runInfo[i] = rSize;
	}
	runInfo[i] = a->size() % rSize;
	if (a->size() % rSize == 0)
	{
		runInfo[i] = rSize;
	}
	int pos = 0;

	auto start = a->begin();
	auto end = start + pos;

	for ( i = 0; i < runInfo.size(); i++)
	{
		start = end;
		end = start + runInfo[i];
		sort(start, end);
	}
	return runInfo;
}

void showRuns(vector<int>* a, vector<int>* runInfo)
{
	int pos = 0;
	cout << "\n串的信息：\n";
	for (int runCount = 0; runCount < runInfo->size(); runCount++)
	{
		cout << "串" << runCount << "：\n";
		for (int j = 0; j < runInfo->at(runCount); j++)
		{
			cout << a->at(pos)<<"  ";
			++pos;
		}
		cout << endl;
	}
}

void mergeSort(vector<int>* a, vector<int>* runInfo)
{
	int runNumber = runInfo->size();
	if (runNumber == 1)
	{//归并到只有一个串了，即归并完成
		return;
	}
	int newRunNumber = (runNumber - 1) / 2 + 1;
	int ni = 0;
	for (int i = 1; i < runNumber;)
	{
		merge(a, runInfo, i - 1, i);
		++ni;
		i += 2;
	}
	if (ni == newRunNumber - 1)
	{
		auto rif = runInfo->begin();
		rif += ni;
		
		*rif = runInfo->at(runNumber - 1);
	}
	runInfo->resize(newRunNumber);
	mergeSort(a, runInfo);
}

void merge(vector<int>* a, vector<int>* runInfo, int run0, int run1)
{
	Buffer input0(bufferSize);
	Buffer input1(bufferSize);
	Buffer output(bufferSize);

	
	int vA = 0;
	int vB = 0;

	int pos0 = 0;
	for (int i = 0; i < run0 / 2; ++i)
	{
		pos0 += runInfo->at(i);
		
	}
	int pos1 = pos0 + runInfo->at(run0);
	int end0 = pos0 + runInfo->at(run0);
	int end1 = pos1 + runInfo->at(run1);


	/*cout << "runNumber:" << runInfo->size() << endl;
	cout << "pos0 end0:" << pos0 << " " << end0 << endl;
	cout << "pos1 end1:" << pos1 << " " << end1 << endl;*/

	auto posA = a->begin();
	int pA = 0;
	pA += pos0;

	posA += pos0;

	input0.writeBuffer(a, pos0, bufferSize);
	input1.writeBuffer(a, pos1, bufferSize);

	while (pos0 < end0 && pos1 < end1)
	{
		if (input0.isInBuffer(pos0))
		{
			vA = input0.readBuffer(pos0);
		}
		else
		{
			input0.clearBuffer();
			input0.writeBuffer(a, pos0, bufferSize);
			vA = input0.readBuffer(pos0);
		}
		if (input1.isInBuffer(pos1))
		{
			vB = input1.readBuffer(pos1);
		}
		else
		{
			input1.clearBuffer();
			input1.writeBuffer(a, pos1, bufferSize);
			vB = input1.readBuffer(pos1);
		}

		if (vA < vB)
		{
			if (!output.getIsFull())
			{
				output.writeBuffer(vA);
			}
			else
			{
				vector<int> aa = output.getBuffer();
				output.clearBuffer();
				for (int i = 0; i < aa.size(); i++)
				{
					*posA = aa[i];
					posA++;
				}
				output.writeBuffer(vA);
			}
			++pos0;
		}
		else
		{
			if (!output.getIsFull())
			{
				output.writeBuffer(vB);
			}
			else
			{
				vector<int> aa = output.getBuffer();
				output.clearBuffer();
				for (int i = 0; i < aa.size(); i++)
				{
					*posA = aa[i];
					posA++;
				}
				output.writeBuffer(vB);
			}
			++pos1;
		}

		

	}

	if (!output.getIsEmpty())
	{
		vector<int> aa = output.getBuffer();
		for (int i = 0; i < output.getLength(); i++)
		{
			*posA = aa[i];
			posA++;
			pA++;
		}
		output.clearBuffer();
	}

	while (pos0 < end0)
	{
		*posA = a->at(pos0);
		posA++;
		pA++;
		pos0++;
	}

	while (pos1 < end1)
	{
		*posA = a->at(pos1);
		posA++;
		pA++;
		pos1++;
	}
	auto rif = runInfo->begin();
	rif += run0 / 2;
	*rif = runInfo->at(run0) + runInfo->at(run1);
}

void externalMergeSort(int length, int runSize)
{
	cout << "当前待排序的文件大小为" << length << "\n串的大小为" << runSize << endl;
	//1. Run generation
	vector<int> a = initData(length);
	cout << "待排序的文件内容为：\n";
	for (int i = 0; i < a.size(); i++)
	{
		cout << a[i] << " ";
	}
	cout <<"\n-------------------------------" << endl;

	vector<int> runInfo = runGeneration(&a, runSize);

	cout << "排序好的串为:\n";
	showRuns(&a, &runInfo);
	cout << "\n-------------------------------" << endl;
	//2. Run merging
	cout << "排序完成！\n";
	mergeSort(&a, &runInfo);
	cout << "归并排序完成后的结果：\n";
	for (int i = 0; i < a.size(); i++)
	{
		cout << a[i] << "  ";
	}
	cout << endl;
}

