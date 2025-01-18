#pragma once
#include"Cache.h"
#include<fstream>
using namespace std;
class CachePool {
public:
	Cache* cachePool;//缓存区数组
	int cacheCapacity;//缓存区容量
	int cacheNum;//缓存区个数，根据K来决定
	int waitCacheID;//空闲缓存区的下标
	int* WayIDtoCacheID;//败者树的路和其读取的缓存区下标的映射数组，即可通过路的下标找到其对应缓存区的下标
	bool* finish;//保存每个缓存区是否已完成读取的数组

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