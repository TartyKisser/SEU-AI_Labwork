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


//��ջ����
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


//�Ӷ�Ӧ·�Ļ�������ȡ����
int CachePool::readFromCache(int WayID)
{
	int CacheID = WayIDtoCacheID[WayID];
	return cachePool[CacheID].getCurrent();
}

//�鿴���л�������ĩβ���ݵ�ֵ���ҳ���ֵ��С����һ·����һ·����챻����
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


//Ϊĳһ�������������ݣ����WIDȱʡ�����ʾΪ���л�������������
void CachePool::loadCache(fstream& file, int diskStart, int diskEnd, int WID)
{
	int CacheID = 0;
	CacheID = (WID == -1) ? waitCacheID : (WayIDtoCacheID[WID]);
	cachePool[CacheID].clear();

	//������λ�ò�����diskEnd����������������������ʱ����������
	file.seekg(diskStart * sizeof(int), ios::beg);
	for (int i = diskStart; i < diskEnd && (cachePool[CacheID].size < cacheCapacity); i++)
	{
		file.read((char*)&cachePool[CacheID].cache[cachePool[CacheID].size], sizeof(int));
		cachePool[CacheID].size++;
	}
	cachePool[CacheID].diskStartPosition = diskStart;
	cachePool[CacheID].diskEndPosition = diskEnd;

	//�����������̿�ʼλ�ü��϶��뻺������ݸ������ڴ��̽���λ��ʱ����ʾ����ɹ鲢���������ݵĶ�ȡ��
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

//�鿴��Ӧ·�µĻ������Ƿ�����ɶ�ȡ
bool CachePool::isFinish(int WayID)
{
	return finish[WayIDtoCacheID[WayID]];
}

//ͳ������ɻ����ȡ�Ļ�������������finish=K��ʱ���ʾ�����������
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

//����Ӧ·�Ļ��������Ϊ�����
void CachePool::setFinish(int WayID)
{
	finish[WayIDtoCacheID[WayID]] = true;
}

//�鿴��Ӧ·��д�뻺�����ĵ�ǰ�鲢�������Ƿ��ѷ�������
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

//ʹ�ÿ��л���Ϊ��Ӧ·�µĻ�������������
void CachePool::loadCredit(int WayID)
{
	//����Ӧ·��WayID����Ԥ���صĿ��л���CacheID�󶨣�����ԭ���Ļ�����գ���Ϊ�µĿ��л��棬����Ԥ��������
	int CacheID = WayIDtoCacheID[WayID];
	WayIDtoCacheID[WayID] = waitCacheID;
	cachePool[CacheID].clear();
	waitCacheID = CacheID;
}
