#include"Cache.h"
#define min(x,y) (x<=y?x:y)
Cache::Cache()
{
	capacity = 50;
	cache = new int[capacity];
	for (int i = 0;i < capacity;i++) {
		cache[i] = 0;
	}
	//firstPosition = 0;
	//lastPosition = 0;
	diskStartPosition = -1;
	diskEndPosition = -1;
	size = 0;
	currentCachePosition = 0;
}
Cache::Cache(int maxSize)
{
	capacity = maxSize;
	cache = new int[capacity];
	for (int i = 0;i < capacity;i++) {
		cache[i] = 0;
	}
	//firstPosition = 0;
	//lastPosition = 0;
	diskStartPosition = -1;
	diskEndPosition = -1;
	size = 0;
	currentCachePosition = 0;
}

void Cache::clear()
{
	for (int i = 0;i < size;i++) {
		cache[i] = 0;
	}
	//firstPosition=0;
	//lastPosition=0;
	diskStartPosition = -1;
	diskEndPosition = -1;
	size = 0;
	currentCachePosition = 0;
}

void Cache::add(int value)
{
	if (size >= capacity)
		throw "Cache full!";
	else {
		cache[size] = value;
		size++;
		//lastPosition++;
	}
}

bool Cache::isFull()
{
	return size == capacity;
}

bool Cache::isEmpty()
{
	return size == 0;
}

int Cache::getCurrent()
{
	currentCachePosition++;
	return cache[currentCachePosition - 1];
}