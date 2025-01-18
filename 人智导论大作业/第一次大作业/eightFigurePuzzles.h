#pragma once
#include <iostream>
#include <vector>
using namespace std;
const int puzzleNum = 8;
/*
 *描述：记录8数码九宫格内每一个格子对应的位置信息与存放的数码。
 * xPosition :行数 0 代表第一行
 * yPosition: 列数 0 代表第一行
 * puzzleId: 数码 0~8
 */
typedef struct
{
  int xPosition;
  int yPosition;
  int puzzleId;
} PUZZLE;

/*
*描述：声明一个节点，存储当前九宫格状态
*      以目标状态为例，puzzle: {{0,0,0},{0,1,1},{0,2,2},
         {1,0,3},{1,1,4},{1,2,5},
         {2,0,6},{2,1,7},{2,2,8},}

      nextActionList: {[1,0]    向上移动
           [-1,0]   向下移动
           [0,1]    向左移动
           [0,-1]}  向右移动

      nextAxtionList的大小<=4

      depth： 当前状态所处深度
*/
typedef struct
{
  // vector<PUZZLE> puzzles;
  PUZZLE puzzle[9];
  vector<vector<int> > nextActionList;
  vector<vector<int> > precedeActionList;
  int depth;
} PUZZLE_NODE;

/*
 * 输入：节点状态puzzleNode
 * 输出  空格位置，二维数组。
 * 描述：找到 空格 0 所在的位置,返回1个2维数组，分别代表行数和列数;
 */
int *findZeroPosition(PUZZLE_NODE puzzleNode);

/*
 *
 * 输入：节点状态puzzleNode
 * 输出：actionList初始化后的puzzleNode
 *描述：更新puzzleNode的后继可操作动作状态，其中 （1，0）代表空格向上移动，（-1，0）代表空格向下移动，（0，1）代表空格向左移动，（0，-1）代表空格向右移动。
 */
PUZZLE_NODE updatePuzzleNodeActionList(PUZZLE_NODE puzzleNode);

/*
 * 输入：给定动作数组，例如[1,0],代表空格向上移动
 * 输出：puzzleNode在执行完输入动作后得到的新的数码状态
 * 描述：给定动作action（action为二维数组）和puzzleNode,返回执行该动作后新的节点
 */
PUZZLE_NODE moveToPuzzleNode(vector<int> action, PUZZLE_NODE puzzleNode);

/*
 * 输入：puzzleNode的后继动作数量的大小
 * 输出：随机动作索引
 * 描述：用于生成PuzzleNode中随机动作索引，用于随机后退。
 */
int getRandomNumber(int actionSize);

/*
 * 输入：puzzle1:当前状态某一位置上的状态，puzzle2：目标状态某一位置上的状态
 * 输出：true:相等 false:不相等
 * 描述：判断当前节点状态和目标节点状态在同一位置上两个8数码状态是否相同。
 */
bool isEqual(PUZZLE puzzle1, PUZZLE puzzle2);

/*
 * 输入：当前数码节点状态currentNode，目标数码节点状态objNode
 * 输出：两个节点状态是否匹配，如果匹配，说明找到目标状态，返回true;
 *                             如果不匹配，说明还未找到目标状态，返回false;
 *描述：检测当前节点和目标节点状态是否相同。
 */
bool checkObject(PUZZLE_NODE currentNode, PUZZLE_NODE objNode);

/*
 * 输入：回退步数
 * 输出：给定目标状态回退backwardSteps后的初始状态
 * 描述：给定回退步数，返回初始节点状态
 */
PUZZLE_NODE initialPuzzleNode(int backwardSteps);

/*
 * 输入：动作
 * 输出：给定目标状态回退backwardSteps后的初始状态
 * 描述：输出动作
 */
void outputAction(vector<int> action, int index);

// 用于生成当前状态对应的唯一数字，用于eightFigureFramework中visited判断当前节点状态是否访问过。
int visitedNum(PUZZLE_NODE puzzleNode);

#include <time.h>
#include "eightFigurePuzzles.h"

/*
 *输入： 当前节点状态
 *输出： 当前状态下移动之后不正确位置的数码个数最少的动作
 *描述：给出当前节点状态，根据错位数判断动作
 */
vector<int> InformedByIncorrectNum(PUZZLE_NODE puzzleNode, PUZZLE_NODE objPuzzleNode, map<int, int> visited);

/*
 *输入： 当前节点状态
 *输出： 当前状态下移动之后曼哈顿距离最少的动作
 *描述：给出当前节点状态，根据曼哈顿距离判断动作
 */
vector<int> InformedByManhattonDis(PUZZLE_NODE puzzleNode, PUZZLE_NODE objPuzzleNode, map<int, int> visited);

//----------------------------------------------------------------

// 找到 空格 0 所在的位置
int *findZeroPosition(PUZZLE_NODE puzzleNode)
{
  int res[2] = {0, 0};
  for (int i = 0; i < puzzleNum + 1; i++)
  {
    if (puzzleNode.puzzle[i].puzzleId == 0)
    {
      res[0] = puzzleNode.puzzle[i].xPosition;
      res[1] = puzzleNode.puzzle[i].yPosition;
      return res;
    }
  }
  return res;
}

// 更新puzzleNode的后继可操作动作状态，其中 （1，0）代表空格向上移动，（-1，0）代表空格向下移动，（0，1）代表空格向左移动，（0，-1）代表空格向右移动。
PUZZLE_NODE updatePuzzleNodeActionList(PUZZLE_NODE puzzleNode)
{
  puzzleNode.nextActionList.clear();
  int *xyPosition = findZeroPosition(puzzleNode);
  int x = xyPosition[0];
  int y = xyPosition[1];
  if (x >= 1)
  {
    vector<int> actionUp;
    actionUp.push_back(1);
    actionUp.push_back(0);
    puzzleNode.nextActionList.push_back(actionUp);
  }
  if (x <= 1)
  {
    vector<int> actionDown;
    actionDown.push_back(-1);
    actionDown.push_back(0);
    puzzleNode.nextActionList.push_back(actionDown);
  }
  if (y >= 1)
  {
    vector<int> actionLeft;
    actionLeft.push_back(0);
    actionLeft.push_back(1);
    puzzleNode.nextActionList.push_back(actionLeft);
  }
  if (y <= 1)
  {
    vector<int> actionRight;
    actionRight.push_back(0);
    actionRight.push_back(-1);
    puzzleNode.nextActionList.push_back(actionRight);
  }
  return puzzleNode;
}

void outputAction(vector<int> action, int index)
{
  /*cout << action[0] << " " << action[1] << endl;*/
  if (action[0] == 1 && action[1] == 0)
  {
    cout << "步数 " << index << ":向上移动" << endl;
    cout << endl;
  }
  else if (action[0] == -1 && action[1] == 0)
  {
    cout << "步数 " << index << "向下移动" << endl;
    cout << endl;
  }
  else if (action[0] == 0 && action[1] == 1)
  {
    cout << "步数 " << index << "向左移动" << endl;
    cout << endl;
  }
  else
  {
    cout << "步数 " << index << "向右移动" << endl;
    cout << endl;
  }
}

// 给定动作action（action为二维数组）和puzzleNode,返回执行该动作后新的节点
PUZZLE_NODE moveToPuzzleNode(vector<int> action, PUZZLE_NODE puzzleNode)
{
  // cout << action[0] << " " << action[1] << endl;
  // if (action[0] == 1 && action[1] == 0) {
  //  cout << "向上移动" << endl;
  // }
  // else if (action[0] == -1 && action[1] == 0) {
  //  cout << "向下移动" << endl;
  // }
  // else if (action[0] == 0 && action[1] == 1) {
  //  cout << "向左移动" << endl;
  // }
  // else {
  //  cout << "向右移动" << endl;
  // }

  int *xyPosition = findZeroPosition(puzzleNode);
  int x = xyPosition[0];
  int y = xyPosition[1];
  PUZZLE_NODE nextPuzzleNode;
  for (int xPos = 0; xPos < 3; xPos++)
  {
    for (int yPos = 0; yPos < 3; yPos++)
    {
      nextPuzzleNode.puzzle[xPos * 3 + yPos].xPosition = xPos;
      nextPuzzleNode.puzzle[xPos * 3 + yPos].yPosition = yPos;
      if (xPos == x && yPos == y)
      {
        nextPuzzleNode.puzzle[xPos * 3 + yPos].puzzleId = puzzleNode.puzzle[((x - action[0]) * 3 + (y - action[1]))].puzzleId;
      }
      else if (xPos == (x - action[0]) && yPos == (y - action[1]))
      {
        nextPuzzleNode.puzzle[xPos * 3 + yPos].puzzleId = puzzleNode.puzzle[(x * 3 + y)].puzzleId;
      }
      else
      {
        nextPuzzleNode.puzzle[xPos * 3 + yPos].puzzleId = puzzleNode.puzzle[xPos * 3 + yPos].puzzleId;
      }
    }
  }
  return nextPuzzleNode;
}

// 用于生成PuzzleNode中随机动作索引
int getRandomNumber(int actionSize)
{
  int RandomNumber;
         // time()用系统时间初始化种。为rand()生成不同的随机种子。
  RandomNumber = rand() % actionSize; // 生成1~100随机数
  return RandomNumber;
}

// 给定回退步数，返回初始状态
PUZZLE_NODE initialPuzzleNode(int backwordSteps)
{
  PUZZLE_NODE objNode;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      objNode.puzzle[i * 3 + j].puzzleId = i * 3 + j;
      objNode.puzzle[i * 3 + j].xPosition = i;
      objNode.puzzle[i * 3 + j].yPosition = j;
    }
  }
  PUZZLE_NODE initialPuzzleNode = updatePuzzleNodeActionList(objNode);

  for (int i = 0; i < backwordSteps; i++)
  {
    PUZZLE_NODE precedePuzzleNode = initialPuzzleNode;
    int action = getRandomNumber(initialPuzzleNode.nextActionList.size());
    initialPuzzleNode = moveToPuzzleNode(initialPuzzleNode.nextActionList[action], initialPuzzleNode);
    initialPuzzleNode = updatePuzzleNodeActionList(initialPuzzleNode);
  }
  initialPuzzleNode = updatePuzzleNodeActionList(initialPuzzleNode);
  return initialPuzzleNode;
}

// 判断两个8数码状态是否相同
bool isEqual(PUZZLE puzzle1, PUZZLE puzzle2)
{
  if (puzzle1.xPosition == puzzle2.xPosition && puzzle1.yPosition == puzzle2.yPosition && puzzle1.puzzleId == puzzle2.puzzleId)
    return true;
  else
    return false;
}

// 检测当前节点和目标节点状态是否相同
bool checkObject(PUZZLE_NODE currentNode, PUZZLE_NODE objNode)
{
  for (int i = 0; i < puzzleNum + 1; i++)
  {
    if (!isEqual(currentNode.puzzle[i], objNode.puzzle[i]))
      return false;
  }
  return true;
}

// 判断当前节点状态是否被访问过。
int visitedNum(PUZZLE_NODE puzzleNode)
{
  int mapValue = 0;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      mapValue = mapValue * 10 + puzzleNode.puzzle[i * 3 + j].puzzleId;
    }
  }
  return mapValue;
}

// 给出当前节点状态，根据错位数判断动作
vector<int> InformedByIncorrectNum(PUZZLE_NODE puzzleNode, PUZZLE_NODE objPuzzleNode, map<int, int> visited)
{
  puzzleNode = updatePuzzleNodeActionList(puzzleNode);
  int actionNum = puzzleNode.nextActionList.size();
  int *jugleArray = new int[actionNum];
  // 计算当前节点状态的错位数
  int currentNum = 0;
  for (int i = 0; i < 9; i++)
  {
    if (isEqual(puzzleNode.puzzle[i], objPuzzleNode.puzzle[i]))
      currentNum++;
  }
  // 计算各个动作产生的错位数差值
  for (int i = 0; i < actionNum; i++)
  {
    PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(puzzleNode.nextActionList[i], puzzleNode);
    int nextNum = 0;
    for (int j = 0; j < 9; j++)
    {
      if (isEqual(nextPuzzleNode.puzzle[j], objPuzzleNode.puzzle[j]))
        nextNum++;
    }
    jugleArray[i] = nextNum - currentNum;
  }
  // 比较哪一个最优
  int k=-1;
  int temp = -9;
  for (int i = 0; i < actionNum; i++)
  {
    PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(puzzleNode.nextActionList[i],puzzleNode);
    if (temp < jugleArray[i]&&!visited[visitedNum(nextPuzzleNode)])
    {
      temp = jugleArray[i];
      k = i;
    }
  }
  if(k==-1)
  {
    k = getRandomNumber(puzzleNode.nextActionList.size());
  }
  return puzzleNode.nextActionList[k];
}

// 给出当前节点状态，根据曼哈顿距离判断动作
vector<int> InformedByManhattonDis(PUZZLE_NODE puzzleNode, PUZZLE_NODE objPuzzleNode, map<int, int> visited)
{
  puzzleNode = updatePuzzleNodeActionList(puzzleNode);
  int actionNum = puzzleNode.nextActionList.size();
  int *jugleArray = new int[actionNum];
  // 计算当前节点状态的曼哈顿距离
  int currentDis = 0;
  for (int i = 0; i < 9; i++)
  {
    for (int j = 0; j < 9; j++)
    {
      if (puzzleNode.puzzle[i].puzzleId == objPuzzleNode.puzzle[j].puzzleId)
      {
        currentDis += abs(puzzleNode.puzzle[i].xPosition - objPuzzleNode.puzzle[j].xPosition) + abs(puzzleNode.puzzle[i].yPosition - objPuzzleNode.puzzle[j].yPosition);
      }
    }
  }
  // 计算各个动作产生的曼哈顿距离
  for (int c = 0; c < actionNum; c++)
  {
    PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(puzzleNode.nextActionList[c], puzzleNode);
    int nextDis = 0;
    for (int i = 0; i < 9; i++)
    {
      for (int j = 0; j < 9; j++)
      {
        if (nextPuzzleNode.puzzle[i].puzzleId == objPuzzleNode.puzzle[j].puzzleId)
        {
          nextDis += abs(nextPuzzleNode.puzzle[i].xPosition - objPuzzleNode.puzzle[j].xPosition) + abs(nextPuzzleNode.puzzle[i].yPosition - objPuzzleNode.puzzle[j].yPosition);
        }
      }
    }
    jugleArray[c] = currentDis - nextDis;
  }
  // 比较哪一个最优
  int k=-1;
  int temp = -50;
  for (int i = 0; i < actionNum; i++)
  {
    PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(puzzleNode.nextActionList[i],puzzleNode);
    if (temp < jugleArray[i]&&!visited[visitedNum(nextPuzzleNode)])
    {
      temp = jugleArray[i];
      k = i;
    }
  }
  if(k==-1)
  {
    k = getRandomNumber(puzzleNode.nextActionList.size());
  }
  return puzzleNode.nextActionList[k];
}