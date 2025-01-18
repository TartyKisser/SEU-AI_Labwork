#pragma once
#include<vector>
#include<iostream>
#include<queue>
using namespace std;
template <class T>
class DoubleEndPriorityQueue
{
public:
    //最大堆
    priority_queue<int, vector<int>, greater<int>>* acc;
    //最小堆
    priority_queue<int, vector<int>, less<int>>* desc;
    int size()
    {
        return acc->size();
    }

    bool isEmpty()
    {
        return desc->empty();
    }

    void insert(int x)
    {
        acc->push(x);
        desc->push(x);
    }

    int getMin()
    {
        return acc->top();
    }

    int getMax()
    {
        return desc->top();
    }

    int minPop()
    {
        int min = acc->top();
        bool flag = false;
        acc->pop();
        priority_queue<int, vector<int>, less<int>>* temp = new priority_queue<int, vector<int>, less<int>>();
        while (!desc->empty())
        {
            if (desc->top() != min)
            {
                temp->push(desc->top());
                desc->pop();
            }
            else
            {
                if (!flag)
                {
                    flag = true;
                    desc->pop();
                }
                else
                {
                    temp->push(desc->top());
                    desc->pop();
                }
            }
        }
        delete desc;
        desc = temp;
        return min;
    }

    void deleteMax()
    {
        int i = desc->top();
        bool flag = false;
        desc->pop();
        priority_queue<int, vector<int>, greater<int>>* temp = new priority_queue<int, vector<int>, greater<int>>();
        while (!acc->empty())
        {
            if (acc->top() != i)
            {
                temp->push(acc->top());
                acc->pop();
            }
            else
            {
                if (!flag)
                {
                    flag = true;
                    acc->pop();
                }
                else
                {
                    temp->push(acc->top());
                    acc->pop();
                }
            }
        }
        delete acc;
        acc = temp;
    }

    DoubleEndPriorityQueue()
    {
        acc = new priority_queue<int, vector<int>, greater<int>>();
        desc = new priority_queue<int, vector<int>, less<int>>();
    }
    DoubleEndPriorityQueue(const DoubleEndPriorityQueue<T>& right)
    {
        acc = right.acc;
        desc = right.desc;
    }
};
