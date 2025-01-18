#include"Memory.h"
#include<iostream>
#include<thread>
#include<mutex>
#include<algorithm>
using namespace std;
#define MAXDATA 99999;
#define min(x,y) (x<=y?x:y)
#define tempFileString "tempFile.dat"

mutex output0_mutex;
mutex output1_mutex;
mutex outputDisk_mutex;
mutex input_mutex;
bool input_ready = false;
bool output0_full = false;
bool output1_full = false;
bool mergeFinish = false;
bool diskInFinish = false;


void Show(string filePath) {
    fstream file;
    file.open(filePath, ios::binary | ios::in);
    int temp = 0;
    while (file.peek() != EOF) {
        file.read((char*)&temp, sizeof(int));
        cout << temp << "		";
    }
    cout << endl;
}


//将数据文件处理为有序的最小归并段，用于后续归并过程
void Memory::innerSort()
{
    file.open(filePath, ios_base::out | ios::out | ios::in | ios::binary);
    for (int i = 0; i < dataLength; i += cacheCapacity)
    {
        file.seekg(i * sizeof(int), ios::beg);
        output0.clear();
        for (int j = 0;j < output0.capacity && (i + j) < dataLength;j++) {
            file.read((char*)&output0.cache[output0.size], sizeof(int));
            //cout << input0.cache[j] << " ";
            output0.size++;
        }
        IOCount++;
        sort(output0.cache, output0.cache + output0.size);
        file.seekg(i * sizeof(int), ios::beg);
        for (int k = 0;k < output0.capacity && (i + k) < dataLength;k++) {
            //cout << input0.cache[k] << " ";
            file.write((char*)&output0.cache[k], sizeof(int));
            output0.size--;
        }
        IOCount++;
    }
    file.close();
}

Memory::Memory(int Capacity, string fileString, int fileLength, int k)
{
    cacheCapacity = Capacity;
    output0 = Cache(cacheCapacity);
    output1 = Cache(cacheCapacity);
    K = k;
    inputCachePool = CachePool(K, cacheCapacity);
    filePath = fileString;
    tempFilePath = tempFileString;
    dataLength = fileLength;
}


//k路归并排序的实现
void Memory::kWayMergeSort()
{
    //cout << endl << "Init File:" << endl;
    //Show(filePath);

    innerSort();

    //磁盘归并段长度依次翻倍递增，直至整个磁盘数据排序完成
    for (int step = cacheCapacity * K; step / K < dataLength; step *= K)
    {
        file.open(filePath, ios_base::out | ios::out | ios::in | ios::binary);

        //清空用于保存归并结果的临时文件
        remove(tempFilePath.c_str());
        output0.clear();
        output1.clear();
        inputCachePool.clear();

        for (int i = 0; i < dataLength; i += step)
        {
            //minPosition为最左划分位置
            int minPosition = i + step / K;
            if (minPosition < dataLength) {
                kWayMerge(i, i+step);
            }
            else {
                writeBack(i, dataLength);
            }
        }
        file.close();
        remove(filePath.c_str());
        rename(tempFilePath.c_str(), filePath.c_str());
    }
    cout << "KWayMergeSort Finish------Data Length:" << dataLength << "		" 
        << "cacheCapacity:" << cacheCapacity << "		" << "I/O_Count:" << IOCount << "\n";
}


//归并的最小操作，对k个归并段进行归并
void Memory::kWayMerge(int diskStart, int diskEnd)
{
    int start = diskStart;
    int end = diskEnd;

    //out0_thread负责output0的磁盘写，out1_thread负责output1的磁盘写，merge_thread负责基于败者树的归并过程，in_thread负责input缓存池的预加载（磁盘读）
    thread out0_thread = thread(&Memory::writeIntoDisk,this, false);
    thread out1_thread = thread(&Memory::writeIntoDisk,this, true);
    thread merge_thread = thread(&Memory::runMerge,this, ref(start), ref(end));
    thread in_thread = thread(&Memory::readFromDisk,this);

    out0_thread.join();
    out1_thread.join();
    merge_thread.join();
    in_thread.join();
}


//将不满足最小归并长度需求的数据段直接写回原始文件
void Memory::writeBack(int diskStart, int diskEnd)
{
    int temp = 0;
    fstream fout(tempFilePath, ios_base::out | ios::binary | ios::app);
    //如果起始位置大于终止位置，则抛出异常
    if (diskStart > diskEnd)
    {
        throw invalid_argument("diskStart > diskEnd");
        exit(1);
    }

    fout.seekg(diskStart * sizeof(int), ios::beg);
    file.seekg(diskStart * sizeof(int), ios::beg);

    for (int i = diskStart; i < diskEnd; i++)
    {
        file.read((char*)&temp, sizeof(int));
        fout.write((char*)&temp, sizeof(int));
    }
    fout.close();
}

//从磁盘中读取数据存入input缓存池的空闲缓存
void Memory::readFromDisk()
{
    //当k路的归并段还未全部读取进input时，仍需要预加载
    while (!diskInFinish) {
        //不断查看空闲缓存的状态，当其已完成加载时在这里阻塞
        while (input_ready);
        //进入临界区
        input_mutex.lock();
        //获取预测最先消耗完的归并路
        int creditWay = inputCachePool.credit();
        int diskStart = inputCachePool.getWayCacheDiskStart(creditWay);
        int size = inputCachePool.getWayCacheSize(creditWay);
        //新的磁盘开始位置在上一次input缓存读取的数据的下一位
        int newdiskStart = diskStart + size;
        //新的磁盘结束位置在上一次相同（同一归并段，所以结束的位置时一样的）
        int newdiskEnd= inputCachePool.getWayCacheDiskEnd(creditWay);
        file.seekg(newdiskStart * sizeof(int), ios::beg);
        //调用缓存池的loadCache函数进行数据的加载
        inputCachePool.loadCache(file, newdiskStart, newdiskEnd);
        //空闲缓存已完成加载
        input_ready = true;
        input_mutex.unlock();
    }
}


//两个output缓存共同使用的写磁盘函数，choose=0为output0输出到磁盘，choose=1为output1输出到磁盘
void Memory::writeIntoDisk(bool choose)
{
    ofstream fout(tempFilePath, ios_base::out | ios::binary | ios::app);
    if (!choose) {
        //当归并未完成时，不断查看output0是否已满
        while (!mergeFinish) {
            //当output0为满时，在这里阻塞
            while (!output0_full);

            //进入临界区
            output0_mutex.lock();
            outputDisk_mutex.lock();
            //写磁盘
            for (int i = 0; i < output0.size; i++)
            {
                fout.write((char*)&output0.cache[i], sizeof(int));
            }
            IOCount++;
            output0.clear();
            //output0变为空
            output0_full = false;
            outputDisk_mutex.unlock();
            output0_mutex.unlock();
        }
        //归并完成后，将output0剩余的数据全部写出
        for (int i = 0; i < output0.size; i++)
        {
            fout.write((char*)&output0.cache[i], sizeof(int));
        }
    }
    else {
        while (!mergeFinish) {
            while (!output1_full);
            output0_mutex.lock();
            outputDisk_mutex.lock();
            for (int i = 0; i < output1.size; i++)
            {
                fout.write((char*)&output1.cache[i], sizeof(int));
            }
            IOCount++;
            output1.clear();
            output1_full = false;
            outputDisk_mutex.unlock();
            output0_mutex.unlock();
        }
    }
    fout.close();
}

//初始化线程池，用于败者树最开始的生成（需要提供k个数据才能生成败者树）
void Memory::InitCachePool(int diskStart, int diskEnd)
{
    //计算要归并的数据的总长
    int mergeLength = diskEnd - diskStart;
    //计算每个归并段的长度
    int step = mergeLength / K;
    int WayID = 0;
    //尝试将k个归并段放入内存
    for (int i = diskStart;i < diskEnd;i += step) {
        //当当前归并段的磁盘开始位置i小于数据结束位置时，说明这一路有数据，需要载入缓存
        if (i < dataLength) {
            inputCachePool.loadCache(file, i, min(i + step, dataLength), WayID);
            WayID++;
        }
        //当当前归并段的磁盘开始位置i大于等于数据结束位置时，说明这一路不存在数据，将这一路缓存标记为已完成读取
        else {
            inputCachePool.setFinish(WayID);
        }
    }
}

//基于败者树的归并过程
void Memory::runMerge(int diskStart,int diskEnd) {
    InitCachePool(diskStart, diskEnd);
    int mergeLength = diskEnd - diskStart;
    int count = 0;
    //data保存用于生成败者树的K个数据
    int* data = new int[K];
    //从k路缓存读取数据放入data，用于生成败者树
    for (int i = 0;i < K;i++) {
        //当这一路未读取完成，也即存在归并段数据时，从缓存读取数据
        if (!inputCachePool.isFinish(i)) {
            data[i] = readFromCache(i);
            count++;
        }
        //否则放进一个极大值用于生成败者树(极大值可以帮助判断归并结束：当败者树输出的胜者为MAXDATA时，表示已完成归并)
        else {
            data[i] = MAXDATA;
        }
    }
    losser = LosserTree(data, K);
    //从败者树输出数据到输出缓存，直至归并完成
    while (count < mergeLength) {
        int winner = losser.getWinner();
        int f = MAXDATA;
        //当败者树输出的胜者为MAXDATA时，表示已完成归并
        if (winner == f ) {
            mergeFinish = true;
            break;
        }
        writeIntoOutput(winner);
        //获取被写出的input缓存的那一路，为它补充新的数据
        int newDataWay = losser.getWinnerPosition();
        losser.PopAndAdd(readFromCache(newDataWay));
        count++;
    }
}

//从input缓存池中获取数据，WayID表示读取的是哪一路的缓存
int Memory::readFromCache(int WayID)
{
    //如果对应路下的缓存区已完成读取，则返回一个MAXDATA作为标记数据
    if (inputCachePool.isFinish(WayID)) {
        return MAXDATA;
    }

    //如果对应路下的缓存区的当前归并段数据已被访问完，则需要从空闲缓存载入新数据
    if (inputCachePool.isReadOver(WayID)) {
        while (!input_ready);
        input_mutex.lock();
        inputCachePool.loadCredit(WayID);
        input_mutex.unlock();
    }
    return inputCachePool.readFromCache(WayID);
}

//向输出缓存区写入数据
void Memory::writeIntoOutput(int value)
{
    //当两个输出缓存区都满时，在这里阻塞
    while (output0_full && output1_full);
    //如果output0未满，优先向其写入数据
    if (!output0_full){
        output0_mutex.lock();
        output0.add(value);
        if (output0.isFull()) {
            output0_full = true;
        }
        output0_mutex.unlock();
    }
    //如果output1未满，向output1写入数据
    else {
        output1_mutex.lock();
        output1.add(value);
        if (output1.isFull()) {
            output1_full = true;
        }
        output1_mutex.unlock();
    }
}