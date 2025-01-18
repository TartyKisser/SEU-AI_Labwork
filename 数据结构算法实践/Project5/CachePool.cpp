#include "CachePool.h"

CachePool::CachePool()
{
	cacheNum = 4;
	cacheCapacity = 25;
	cachePool = new Cache[cacheNum];
	WayIDtoCacheID = new int[cacheNum-1];
	finish = new bool[cacheNum];
	for (int i = 0;i < cacheNum;i++) {
		cachePool[i] = Cache(cacheCapacity);
	}
	for (int i = 0;i < cacheNum-1;i++) {
		WayIDtoCacheID[i] = i;
	}
	for (int i = 0;i < cacheNum;i++) {
		finish[i] = false;
	}
	waitCacheID = cacheNum - 1;
}

CachePool::CachePool(int K, int cachecapacity)
{
	cacheNum = K + 1;
	cacheCapacity = cachecapacity;
	cachePool = new Cache[cacheNum];
	WayIDtoCacheID = new int[cacheNum - 1];
	finish = new bool[cacheNum];
	for (int i = 0;i < cacheNum;i++) {
		cachePool[i] = Cache(cacheCapacity);
	}
	for (int i = 0;i < cacheNum - 1;i++) {
		WayIDtoCacheID[i] = i;
	}	
	for (int i = 0;i < cacheNum;i++) {
		finish[i] = false;
	}
	waitCacheID = cacheNum - 1;
}


//清空缓存池
void CachePool::clear() {
	for (int i = 0;i < cacheNum - 1;i++) {
		WayIDtoCacheID[i] = i;
	}
	for (int i = 0;i < cacheNum;i++) {
		finish[i] = false;
	}
	waitCacheID = cacheNum - 1;
	for (int i = 0;i < cacheNum;i++) {
		cachePool[i].clear();
	}
}


//从对应路的缓存区读取数据
int CachePool::readFromCache(int WayID)
{
	int CacheID = WayIDtoCacheID[WayID];
	return cachePool[CacheID].getCurrent();
}

//查看所有缓存区最末尾数据的值，找出其值最小的那一路，这一路将最快被读完
int CachePool::credit()
{
	int min = cachePool[WayIDtoCacheID[0]].cache[cacheCapacity - 1];
	Cache temp;
	int creditWay = 0;
	for (int i = 1;i < cacheNum - 1;i++) {
		temp = cachePool[WayIDtoCacheID[i]];
		if (temp.size > 0&&temp.cache[temp.size-1] < min) {
			creditWay = i;
		}
	}
	return creditWay;
}


//为某一缓存区加载数据，如果WID缺省，则表示为空闲缓存区加载数据
void CachePool::loadCache(fstream& file, int diskStart, int diskEnd, int WID)
{
	int CacheID = 0;
	CacheID = (WID == -1) ? waitCacheID : (WayIDtoCacheID[WID]);
	cachePool[CacheID].clear();

	//当数据位置不超过diskEnd且数据量不超过缓存容量时，读入数据
	file.seekg(diskStart * sizeof(int), ios::beg);
	for (int i = diskStart; i < diskEnd && (cachePool[CacheID].size < cacheCapacity); i++)
	{
		file.read((char*)&cachePool[CacheID].cache[cachePool[CacheID].size], sizeof(int));
		cachePool[CacheID].size++;
	}
	cachePool[CacheID].diskStartPosition = diskStart;
	cachePool[CacheID].diskEndPosition = diskEnd;

	//当缓存区磁盘开始位置加上读入缓存的数据个数等于磁盘结束位置时，表示已完成归并段所有数据的读取。
	if (cachePool[CacheID].size + diskStart == diskEnd) {
		finish[CacheID] = true;
	}
}


int CachePool::getWayCacheDiskStart(int WayID)
{
	int CacheID = WayIDtoCacheID[WayID];
	int diskStart = cachePool[CacheID].diskStartPosition;
	return diskStart;
}

int CachePool::getWayCacheDiskEnd(int WayID)
{
	int CacheID = WayIDtoCacheID[WayID];
	int diskEnd = cachePool[CacheID].diskEndPosition;
	return diskEnd;
}

int CachePool::getWayCacheSize(int WayID)
{
	int CacheID = WayIDtoCacheID[WayID];
	int size = cachePool[CacheID].size;
	return size;
}

//查看对应路下的缓存区是否已完成读取
bool CachePool::isFinish(int WayID)
{
	return finish[WayIDtoCacheID[WayID]];
}

//统计已完成缓存读取的缓存区个数，当finish=K的时候表示读缓存已完成
int CachePool::getFinishNum()
{
	int finishNum = 0;
	for (int i = 0;i < cacheNum;i++) {
		if (finish[i]) {
			finish++;
		}
	}
	return finishNum;
}

//将对应路的缓存区标记为已完成
void CachePool::setFinish(int WayID)
{
	finish[WayIDtoCacheID[WayID]] = true;
}

//查看对应路的写入缓存区的当前归并段数据是否已访问完了
bool CachePool::isReadOver(int WayID)
{
	int CacheID = WayIDtoCacheID[WayID];
	if (cachePool[CacheID].currentCachePosition >= cacheCapacity) {
		return true;
	}
	else {
		return false;
	}
}

//使用空闲缓存为对应路下的缓存区载入数据
void CachePool::loadCredit(int WayID)
{
	//将对应路的WayID与已预加载的空闲缓存CacheID绑定，并把原来的缓存清空，作为新的空闲缓存，用于预加载数据
	int CacheID = WayIDtoCacheID[WayID];
	WayIDtoCacheID[WayID] = waitCacheID;
	cachePool[CacheID].clear();
	waitCacheID = CacheID;
}
