#pragma once
#include<iostream>
class Cache
{
public:
	int* cache;
	int capacity;
	int diskStartPosition;
	int diskEndPosition;
	int size;
	int currentCachePosition;

	Cache();
	Cache(int);
	void clear();
	void add(int);
	bool isFull();
	bool isEmpty();
	int getCurrent();
};