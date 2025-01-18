#pragma once
#include"Cache.h"
#include"CachePool.h"
#include<fstream>
#include<queue>
#include"LosserTree.h"
using namespace std;
class Memory {
public:
	Cache output0;
	Cache output1;

	LosserTree losser;

	CachePool inputCachePool;
	int activeCache;

	fstream file;
	int dataLength;
	int cacheCapacity;
	int K;
	string filePath;
	string tempFilePath;
	int IOCount = 0;

	Memory(int, string, int,int);



	//void Merge(int, int, int, int);
	//void mergeSort();

	void writeBack(int,int);

	void InitCachePool(int,int);


	int readFromCache(int);
	void writeIntoOutput(int);

	void readFromDisk();

	void writeIntoDisk(bool);

	void innerSort();

	void kWayMerge(int,int);
	void kWayMergeSort();

	void runMerge(int,int);
};