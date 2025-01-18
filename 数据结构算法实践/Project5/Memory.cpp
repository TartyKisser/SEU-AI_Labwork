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


//�������ļ�����Ϊ�������С�鲢�Σ����ں����鲢����
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


//k·�鲢�����ʵ��
void Memory::kWayMergeSort()
{
    //cout << endl << "Init File:" << endl;
    //Show(filePath);

    innerSort();

    //���̹鲢�γ������η���������ֱ���������������������
    for (int step = cacheCapacity * K; step / K < dataLength; step *= K)
    {
        file.open(filePath, ios_base::out | ios::out | ios::in | ios::binary);

        //������ڱ���鲢�������ʱ�ļ�
        remove(tempFilePath.c_str());
        output0.clear();
        output1.clear();
        inputCachePool.clear();

        for (int i = 0; i < dataLength; i += step)
        {
            //minPositionΪ���󻮷�λ��
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


//�鲢����С��������k���鲢�ν��й鲢
void Memory::kWayMerge(int diskStart, int diskEnd)
{
    int start = diskStart;
    int end = diskEnd;

    //out0_thread����output0�Ĵ���д��out1_thread����output1�Ĵ���д��merge_thread������ڰ������Ĺ鲢���̣�in_thread����input����ص�Ԥ���أ����̶���
    thread out0_thread = thread(&Memory::writeIntoDisk,this, false);
    thread out1_thread = thread(&Memory::writeIntoDisk,this, true);
    thread merge_thread = thread(&Memory::runMerge,this, ref(start), ref(end));
    thread in_thread = thread(&Memory::readFromDisk,this);

    out0_thread.join();
    out1_thread.join();
    merge_thread.join();
    in_thread.join();
}


//����������С�鲢������������ݶ�ֱ��д��ԭʼ�ļ�
void Memory::writeBack(int diskStart, int diskEnd)
{
    int temp = 0;
    fstream fout(tempFilePath, ios_base::out | ios::binary | ios::app);
    //�����ʼλ�ô�����ֹλ�ã����׳��쳣
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

//�Ӵ����ж�ȡ���ݴ���input����صĿ��л���
void Memory::readFromDisk()
{
    //��k·�Ĺ鲢�λ�δȫ����ȡ��inputʱ������ҪԤ����
    while (!diskInFinish) {
        //���ϲ鿴���л����״̬����������ɼ���ʱ����������
        while (input_ready);
        //�����ٽ���
        input_mutex.lock();
        //��ȡԤ������������Ĺ鲢·
        int creditWay = inputCachePool.credit();
        int diskStart = inputCachePool.getWayCacheDiskStart(creditWay);
        int size = inputCachePool.getWayCacheSize(creditWay);
        //�µĴ��̿�ʼλ������һ��input�����ȡ�����ݵ���һλ
        int newdiskStart = diskStart + size;
        //�µĴ��̽���λ������һ����ͬ��ͬһ�鲢�Σ����Խ�����λ��ʱһ���ģ�
        int newdiskEnd= inputCachePool.getWayCacheDiskEnd(creditWay);
        file.seekg(newdiskStart * sizeof(int), ios::beg);
        //���û���ص�loadCache�����������ݵļ���
        inputCachePool.loadCache(file, newdiskStart, newdiskEnd);
        //���л�������ɼ���
        input_ready = true;
        input_mutex.unlock();
    }
}


//����output���湲ͬʹ�õ�д���̺�����choose=0Ϊoutput0��������̣�choose=1Ϊoutput1���������
void Memory::writeIntoDisk(bool choose)
{
    ofstream fout(tempFilePath, ios_base::out | ios::binary | ios::app);
    if (!choose) {
        //���鲢δ���ʱ�����ϲ鿴output0�Ƿ�����
        while (!mergeFinish) {
            //��output0Ϊ��ʱ������������
            while (!output0_full);

            //�����ٽ���
            output0_mutex.lock();
            outputDisk_mutex.lock();
            //д����
            for (int i = 0; i < output0.size; i++)
            {
                fout.write((char*)&output0.cache[i], sizeof(int));
            }
            IOCount++;
            output0.clear();
            //output0��Ϊ��
            output0_full = false;
            outputDisk_mutex.unlock();
            output0_mutex.unlock();
        }
        //�鲢��ɺ󣬽�output0ʣ�������ȫ��д��
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

//��ʼ���̳߳أ����ڰ������ʼ�����ɣ���Ҫ�ṩk�����ݲ������ɰ�������
void Memory::InitCachePool(int diskStart, int diskEnd)
{
    //����Ҫ�鲢�����ݵ��ܳ�
    int mergeLength = diskEnd - diskStart;
    //����ÿ���鲢�εĳ���
    int step = mergeLength / K;
    int WayID = 0;
    //���Խ�k���鲢�η����ڴ�
    for (int i = diskStart;i < diskEnd;i += step) {
        //����ǰ�鲢�εĴ��̿�ʼλ��iС�����ݽ���λ��ʱ��˵����һ·�����ݣ���Ҫ���뻺��
        if (i < dataLength) {
            inputCachePool.loadCache(file, i, min(i + step, dataLength), WayID);
            WayID++;
        }
        //����ǰ�鲢�εĴ��̿�ʼλ��i���ڵ������ݽ���λ��ʱ��˵����һ·���������ݣ�����һ·������Ϊ����ɶ�ȡ
        else {
            inputCachePool.setFinish(WayID);
        }
    }
}

//���ڰ������Ĺ鲢����
void Memory::runMerge(int diskStart,int diskEnd) {
    InitCachePool(diskStart, diskEnd);
    int mergeLength = diskEnd - diskStart;
    int count = 0;
    //data�����������ɰ�������K������
    int* data = new int[K];
    //��k·�����ȡ���ݷ���data���������ɰ�����
    for (int i = 0;i < K;i++) {
        //����һ·δ��ȡ��ɣ�Ҳ�����ڹ鲢������ʱ���ӻ����ȡ����
        if (!inputCachePool.isFinish(i)) {
            data[i] = readFromCache(i);
            count++;
        }
        //����Ž�һ������ֵ�������ɰ�����(����ֵ���԰����жϹ鲢�������������������ʤ��ΪMAXDATAʱ����ʾ����ɹ鲢)
        else {
            data[i] = MAXDATA;
        }
    }
    losser = LosserTree(data, K);
    //�Ӱ�����������ݵ�������棬ֱ���鲢���
    while (count < mergeLength) {
        int winner = losser.getWinner();
        int f = MAXDATA;
        //�������������ʤ��ΪMAXDATAʱ����ʾ����ɹ鲢
        if (winner == f ) {
            mergeFinish = true;
            break;
        }
        writeIntoOutput(winner);
        //��ȡ��д����input�������һ·��Ϊ�������µ�����
        int newDataWay = losser.getWinnerPosition();
        losser.PopAndAdd(readFromCache(newDataWay));
        count++;
    }
}

//��input������л�ȡ���ݣ�WayID��ʾ��ȡ������һ·�Ļ���
int Memory::readFromCache(int WayID)
{
    //�����Ӧ·�µĻ���������ɶ�ȡ���򷵻�һ��MAXDATA��Ϊ�������
    if (inputCachePool.isFinish(WayID)) {
        return MAXDATA;
    }

    //�����Ӧ·�µĻ������ĵ�ǰ�鲢�������ѱ������꣬����Ҫ�ӿ��л�������������
    if (inputCachePool.isReadOver(WayID)) {
        while (!input_ready);
        input_mutex.lock();
        inputCachePool.loadCredit(WayID);
        input_mutex.unlock();
    }
    return inputCachePool.readFromCache(WayID);
}

//�����������д������
void Memory::writeIntoOutput(int value)
{
    //�������������������ʱ������������
    while (output0_full && output1_full);
    //���output0δ������������д������
    if (!output0_full){
        output0_mutex.lock();
        output0.add(value);
        if (output0.isFull()) {
            output0_full = true;
        }
        output0_mutex.unlock();
    }
    //���output1δ������output1д������
    else {
        output1_mutex.lock();
        output1.add(value);
        if (output1.isFull()) {
            output1_full = true;
        }
        output1_mutex.unlock();
    }
}