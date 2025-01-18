//MergeSortImprove1.cpp
#include <cstdio>
#include<thread>
#include<Windows.h>
#include <mutex>
#include <assert.h>
#include <iostream>
#include <vector>
#include <queue>
#include"Closertree.h"
#include"CBuffer.h"

using namespace std;

const int N = 1000;   // �ļ��к��еļ�¼������
const int TreeN = 32; // ����������

//�����������
vector<int> allData;
//��¼������Ϣ��runInfo.size�����ĸ�����runInfo[0]��ʾ��0�Ĵ�С
vector<int> runInfo;
//��һ��Ҫ�������ݵ�λ��
int posData = 0;
int writeData = 0;

//������
LoserTree losertree(TreeN);
//������
Buffer input1(5);
Buffer input2(5);
Buffer output1(5);
Buffer output2(5);

//���߳����
mutex input_mutex[2];
mutex output_mutex[2];
  //�ж������м��߳����ڶ�ȡ��д���input����
int input_thread_number = 1;
int output_thread_number = 1;
  //������������߳�����ռ�õ�input����
int to_thread_number = 1;
int from_thread_number = 1;
  //��¼����������
int output_sum = 0;
  //��¼���������
int sort_sum = 0;
  //��ǰ��������
int cur_num = 0;


//���ɴ���������
void initData(int size);
//չʾ����������
void showData();

//���ļ������ݵ����뻺����
void threadFunReadFile();
//�����������д���ݵ��ļ�
void threadFunWriteFile();
//�����������뻺�������ļ���д�ļ������������
void threadFunMiddle();

void runGeneration();
void showRuns();
void showBestMergeSequence();

int main() 
{    
    initData(N);
    showData();   
    runGeneration();
    showRuns();
    showBestMergeSequence();
    showData();
    return 0;
}

void initData(int size)
{
    allData.resize(size);
    srand(time(0));
    for (int i = 0; i < size; i++)
    {
        allData[i] = rand() % 100 + 1;
    }
}
void showData()
{
    cout << "\n�����������Ϊ��\n";
    for (int i = 0; i < allData.size(); i++)
    {
        cout << allData[i] << " ";
    }
    cout << endl;
}

void threadFunMiddle()
{
    bool reinit = false;
    while (1)
    {
        //�����Ѿ����꣬���������������������д�����������
        if (posData == N && sort_sum == N)
        {
            while (1)
            {
                if (!reinit)
                {
                    losertree.reInit();
                    reinit = true;
                }
                if (output_thread_number == 1)
                {
                    while (output1.isFull())
                    {

                    }
                    output_mutex[0].lock();
                    if (losertree.getlose() == 99999999)
                    {
                        output_mutex[0].unlock();
                        return;
                    }
                    for (int i = 1; i <= 5; i++)
                    {
                        output1.push(losertree.getlose());
                        losertree.input(99999999);

                    }
                    output_thread_number = 2;
                    output_mutex[0].unlock();
                }
                else
                {
                    while (output2.isFull())
                    {

                    }
                    output_mutex[1].lock();
                    //cout << "\threadFunMiddle lock 1\n";
                    if (losertree.getlose() == 99999999)
                    {
                        output_mutex[1].unlock();
                        //cout << "\threadFunMiddle unlock 1\n";
                        return;
                    }
                    for (int i = 1; i <= 5; i++)
                    {
                        output2.push(losertree.getlose());
                        losertree.input(99999999);
                    }
                    output_thread_number = 1;
                    output_mutex[1].unlock();
                    //cout << "\threadFunMiddle lock 1\n";
                }
            }

        }
        //��û��ʼ��������
        if (sort_sum < TreeN)
        {
            if (input_thread_number == 1)
            {
                while (input1.isEmpty())
                {

                }
                input_mutex[0].lock();

                for (int i = 1; i <= 5; i++)
                {
                    losertree.initInput(input1.get());
                    sort_sum++;
                    if (sort_sum == TreeN)
                    {
                        losertree.reInit();
                    }
                }
                input1.clear();
                input_thread_number = 2;
                input_mutex[0].unlock();
            }
            else {
                while (input2.isEmpty())
                {

                }
                input_mutex[1].lock();

                for (int i = 1; i <= 5; i++)
                {
                    losertree.initInput(input2.get());
                    sort_sum++;
                    if (sort_sum == TreeN)
                    {
                        losertree.reInit();
                    }
                }
                input2.clear();
                input_thread_number = 1;
                input_mutex[1].unlock();
            }
        }
        //�Ѿ���ʼ���ð�����
        else if (sort_sum <= N)
        {
            if (input_thread_number == 1)
            {
                while (input1.isEmpty())
                {

                }
                input_mutex[0].lock();
                if (output_thread_number == 1)
                {
                    while (output1.isFull())
                    {

                    }
                    output_mutex[0].lock();


                    for (int i = 1; i <= 5; i++)
                    {
                        output1.push(losertree.getlose());
                        if (losertree.input(input1.get()))
                        {
                            losertree.reInit();
                        }
                    }
                    output_thread_number = 2;
                    output_mutex[0].unlock();

                }
                else
                {
                    while (output2.isFull())
                    {

                    }
                    output_mutex[1].lock();


                    for (int i = 1; i <= 5; i++)
                    {
                        output2.push(losertree.getlose());
                        if (losertree.input(input1.get()))
                        {
                            losertree.reInit();
                        }
                    }
                    output_thread_number = 1;
                    output_mutex[1].unlock();

                }

                input1.clear();
                input_thread_number = 2;
                input_mutex[0].unlock();
            }
            else
            {
                while (input2.isEmpty())
                {

                }
                input_mutex[1].lock();
                if (output_thread_number == 1)
                {
                    while (output1.isFull())
                    {

                    }
                    output_mutex[0].lock();

                    for (int i = 1; i <= 5; i++)
                    {
                        output1.push(losertree.getlose());
                        if (losertree.input(input2.get()))
                        {
                            losertree.reInit();
                        }
                    }
                    output_thread_number = 2;
                    output_mutex[0].unlock();

                }
                else
                {
                    while (output2.isFull())
                    {

                    }
                    output_mutex[1].lock();


                    for (int i = 1; i <= 5; i++)
                    {
                        output2.push(losertree.getlose());
                        if (losertree.input(input2.get()))
                        {
                            losertree.reInit();
                        }
                    }
                    output_thread_number = 1;
                    output_mutex[1].unlock();


                }

                input2.clear();
                input_thread_number = 1;
                input_mutex[1].unlock();
            }
            sort_sum += 5;
        }
    }
}

void runGeneration()
{
    thread t1(threadFunReadFile);
    thread t2(threadFunMiddle);
    thread t3(threadFunWriteFile);

    t1.join();
    t2.join();
    t3.join();

    cout << "\n��������ɣ���\n";
}

void showRuns()
{
    cout << "\n���ɵĴ�Ϊ��\n" << endl;

    int i = 0;
    for (int runCount = 1; runCount < runInfo.size(); runCount++)
    {
        int t = runInfo[runCount];
        cout << "��" << runCount <<",����СΪ"<<t << ":\n";
        while (t > 0)
        {
            cout<< allData[i] << " ";
            ++i;
            --t;
        }
        cout << endl;
    }
}

void showBestMergeSequence()
{
    cout << "\n��ѹ鲢����\n";
    priority_queue<int, vector<int>, greater<int> > small_heap;
    for (int i = 1; i < runInfo.size(); i++)
    {
        small_heap.emplace(runInfo[i]);
    }

    int a1 = 0;
    int a2 = 0;
    while (small_heap.size() > 1)
    {
        a1 = small_heap.top();
        small_heap.pop();
        a2 = small_heap.top();
        small_heap.pop();
        cout << "��ǰҪ�鲢�Ĵ��Ĵ�СΪ��" << a1 << " " << a2 << endl;
        small_heap.push(a1 + a2);
    }
}

void threadFunWriteFile()
{
    int lastnum = 100000000;
    while (output_sum != N)
    {
        if (from_thread_number == 1)
        {
            while (output1.isEmpty())
            {

            }
            output_mutex[0].lock();

            for (int i = 1; i <= 5; i++)
            {
                if (output1.getlast() < lastnum)
                {                    
                    cur_num++;
                }

                lastnum = output1.get();
                
                if (writeData >= N - 3)
                {
                    lastnum = allData[N - 4];
                }

                

                allData[writeData] = lastnum;
                ++writeData;
                while (runInfo.size() <= cur_num)
                {
                    runInfo.push_back(0);
                }
                runInfo[cur_num]++;
            }

            output1.clear();
            from_thread_number = 2;
            output_mutex[0].unlock();

        }
        else
        {
            while (output2.isEmpty())
            {

            }
            output_mutex[1].lock();


            for (int i = 1; i <= 5; i++)
            {
                if (output2.getlast() < lastnum)
                {
                    cur_num++;                    
                }

                lastnum = output2.get();
                
                if (writeData >= N - 3)
                {
                    lastnum = allData[N - 4];
                }

                

                allData[writeData] = lastnum;
                ++writeData;
                while (runInfo.size() <= cur_num)
                {
                    runInfo.push_back(0);
                }
                runInfo[cur_num]++;
                
            }
            output2.clear();
            from_thread_number = 1;
            output_mutex[1].unlock();

        }

        output_sum += 5;
    }   
}

void threadFunReadFile()
{
    int input_num;
    while (1)
    {
        //�����ļ�β�ͽ���
        if (posData == N)
        {
            return;
        }

        if (to_thread_number == 1)
        {
            while (input1.isFull())
            {
            }
            input_mutex[0].lock();
            for (int i = 1; i <= 5; i++)
            {
                input_num = allData[posData];
                ++posData;
                input1.push(input_num);
            }
            to_thread_number = 2;
            input_mutex[0].unlock();
        }
        else
        {
            while (input2.isFull())
            {

            }
            input_mutex[1].lock();
            for (int i = 1; i <= 5; i++)
            {
                input_num = allData[posData];
                ++posData;
                input2.push(input_num);
            }
            to_thread_number = 1;
            input_mutex[1].unlock();
        }
    }
}