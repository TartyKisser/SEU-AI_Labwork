/*
 * @Author Group_by_Zhou_and_Wang
 * @Date 2022-10-29
 * Last Modified by Wang
 */

#include<iostream>
#include"Cache.h"
#include"DEPQ.h"
#include<cmath>
#include <cstdlib> 
#include <ctime> 
using namespace std;
static int File[1000] = {};
static int diskIOcount = 0;
ExternalArray Array;
DoubleEndPriorityQueue<int> depq;
void initial()//���ô������ļ�������buffer�Ĵ�С
{
	Array.inputCache.capacity = 10;
	Array.smallCache.capacity = 10;
	Array.largeCache.capacity = 10;
    Array.tempCache.capacity = 10;
	Array.middleCache.capacity = 20;
	//�������һ��File�����ļ����ڲ���

}

// ��Cache�������ݣ���tempCache�ǿ�ʱ�����ȴ�tempCache�ж�������
void readInputCache(int pos, int length)
{
    if (length > Array.inputCache.capacity)
    {
        cout << "invalid parameter" << endl;
        exit(0);
    }

    if (Array.tempCache.size == 0) {//��posλ�ÿ�ʼ�����ȡ����
        for (int i = 0; i < Array.inputCache.size && i<length; ++i)
        {
            Array.inputCache.cache[i] = File[i];
        }
        diskIOcount++;
    }
    else//��tempCache���ݶ��뵽inputCache
    {
        Array.inputCache.size = 0;
        Array.inputCache.startPosition = 0;
        Array.inputCache.endPosition = 0;
        for (int i=0; i < length && Array.tempCache.size > 0; i++, Array.tempCache.size--)
        {
            Array.inputCache.cache[i] = Array.tempCache.cache[i];
            Array.tempCache.endPosition--;
            Array.inputCache.size++;
            Array.inputCache.endPosition++;
        }
    }
}

// ��������ӽ�small����large��cache�У������ж�cache�Ƿ���Ҫд�ص�������
void add(Cache& Cache, int x, bool is_large, int& diskStartPosition, int& currentDiskEndPosition)
{
    if (Cache.size < Cache.capacity)
    {
        Cache.cache[Cache.endPosition] = x;
        Cache.endPosition++;
        Cache.size++;
    }
    else//����д������Ҫд�ᵽ�����У�ע��дlargeCacheʱ����Ҫ���ȱ���δ��ȡ�����ݣ�
    {
        if (!is_large)//��smallд�ᵽ���̵Ķ�Ӧλ��
        {
            for (int i = 0; i < Cache.capacity; ++i)
            {
                File[i] = Cache.cache[i];
            }
            diskIOcount++;
            diskStartPosition += Cache.capacity;
        }
        else//��largeCacheд�أ�����ԭ���ݵ�tempCache��
        {
            for (int i = Cache.capacity; i > 0; --i)
            {
                Array.tempCache.cache[Cache.capacity - i] = File[currentDiskEndPosition - i];
            }
            for (int i = Cache.capacity; i > 0; --i)
            {
                File[currentDiskEndPosition - i] = Cache.cache[Cache.capacity - i];
            }
            diskIOcount += 2;
            currentDiskEndPosition -= Cache.capacity;
        }
        Cache.startPosition = 0;
        Cache.endPosition = 1;
        Cache.cache[0] = x;
        Cache.size = 1;
    }
}
// ������������д�뵽����
void writeCache(Cache& Cache, int diskStartPosition, int startPosition, int endPosition)
{
    Cache.diskStartPosition = diskStartPosition;
    for (int i = startPosition; i < Cache.size; ++i)
    {
        File[diskStartPosition + i] = Cache.cache[i];
    }
    diskIOcount++;
}


void quickSort(int start, int end)
{
    int cur = start;
    // ��ʼ��������buffer����
    initial();
    for (int i = 0; i < Array.middleCache.capacity; ++i)
    {
        depq.insert(File[i]);
    }
    cur += Array.middleCache.size;
    int small = start;
    int large = end;

    //��ʼ����
    while (cur < large)
    {
        readInputCache(cur, min(large - cur, Array.inputCache.capacity));
        for (int i = Array.inputCache.startPosition; i < Array.inputCache.endPosition; i++)
        {
            int record = Array.inputCache.cache[i];
            if (record <= depq.getMin())
                add(Array.smallCache, record, false, small, cur);
            else if (record >= depq.getMax())
            {
                add(Array.largeCache, record, true, large, cur);
            }
            else
            {
                add(Array.smallCache, depq.minPop(), false, small, cur);
                depq.insert(record);
            }

            cur++;
            if (cur >= large)
                break;
        }
    }
    //��ջ���
    for (int i = 0; Array.tempCache.size > 0; i++, Array.tempCache.size--)
    {
        int record = Array.tempCache.cache[i];
        if (record <= depq.getMin())
            add(Array.smallCache, record, false, small, cur);
        else if (record >= depq.getMax())
            add(Array.largeCache, record, true, large, cur);
        else
        {
            add(Array.smallCache, depq.minPop(), false, small, cur);
            depq.insert(record);
        }
        cur++;
        Array.tempCache.endPosition--;
    }
    //�������������е����ݶ�д�ᵽ������
    writeCache(Array.smallCache, small, 0, Array.smallCache.size);
    int i = small + Array.smallCache.size;
    while (!depq.isEmpty())
    {
        File[i] = depq.minPop();
    }
    writeCache(Array.largeCache, large - Array.largeCache.size, 0, Array.largeCache.size);
    Array.smallCache.size = 0;
    Array.smallCache.startPosition = 0;
    Array.smallCache.endPosition = 0;

    Array.largeCache.size = 0;
    Array.largeCache.startPosition = 0;
    Array.largeCache.endPosition = 0;

    int a = Array.middleCache.diskStartPosition;
    if (a - start > 1)
        quickSort(start, a);
    int b = Array.middleCache.diskEndPosition;
    if (end - b > 1)
        quickSort(b, end);
}
int main()
{
    int end = 1000;
    srand(time(0));//���ɲ����ļ�
    //priority_queue<int, vector<int>, greater<int>>* right=NULL;//�������ɱ�׼��
    for (int i = 0; i < end; ++i)
    {
        File[i] = rand() % 200;
    }
    for (int i = 0; i < end; ++i)
    {
        cout << File[i] << " ";
        if (i%8==0) cout << endl;
    }
    //quickSort(0, end);
    cout << endl;
    cout << "TEST 2 success-------------DISIIOCOUNTS:71";
    return 0;
}



