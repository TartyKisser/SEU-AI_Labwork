#pragma once
#include"DEPQ.h"
struct Cache
{
    int* cache;
    int capacity;
    // the start position of the cache in disk
    // when the cache is empty, startPosition = endPosition = 0
    int diskStartPosition;
    int diskEndPosition;
    // when Cache is not filled, endPosition - startPosition != capacity
    int startPosition;
    int endPosition;
    int size;
};
class ExternalArray
{
    friend class ExternalArrayTest;

public:
    int length = 0;
    Cache inputCache;
    Cache smallCache;
    Cache largeCache;
    Cache tempCache;
    Cache middleCache;
    int diskIOCount = 0;
};

