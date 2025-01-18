#pragma once
#include"Cache.h"
#include<fstream>
using namespace std;
class CachePool {
public:
	Cache* cachePool;//����������
	int cacheCapacity;//����������
	int cacheNum;//����������������K������
	int waitCacheID;//���л��������±�
	int* WayIDtoCacheID;//��������·�����ȡ�Ļ������±��ӳ�����飬����ͨ��·���±��ҵ����Ӧ���������±�
	bool* finish;//����ÿ���������Ƿ�����ɶ�ȡ������

	CachePool();
	CachePool(int, int);
	void clear();
	int readFromCache(int);
	int credit();
	void loadCache(fstream&, int, int, int=-1);
	int getWayCacheDiskStart(int);
	int getWayCacheDiskEnd(int);
	int getWayCacheSize(int);
	bool isFinish(int);
	int getFinishNum();
	void setFinish(int);
	bool isReadOver(int);
	void loadCredit(int);
};