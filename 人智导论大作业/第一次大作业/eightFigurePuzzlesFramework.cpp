#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <map>
#include <time.h>
#include "eightFigurePuzzles.h"
using namespace std;

// 用于记录当前状态是否被访问过。
map<int, int> visited;

// 深度优先搜索，用于限制深度。
#define MAX_DEPTH 20

// openList与closeList用于A*搜索。
vector<PUZZLE_NODE> closeList;
vector<PUZZLE_NODE> openList;

// 广度优先搜索
int *binaryFirstSearch(PUZZLE_NODE initialNode, PUZZLE_NODE objPuzzleNode)
{
  // result[0] 1:correct;0:wrong
  // result[1] 步数 steps
  int result[2] = {0, 0};

  cout << "初始节点状态：" << endl;
  for (int i = 0; i < 3; i++)
  {
    cout << " " << initialNode.puzzle[i * 3 + 0].puzzleId << "  " << initialNode.puzzle[i * 3 + 1].puzzleId << "  " << initialNode.puzzle[i * 3 + 2].puzzleId << endl;
  }
  cout << endl;
  PUZZLE_NODE puzzleNode = initialNode;
  queue<PUZZLE_NODE> puzzleNodeQueue;
  puzzleNode.depth = 0;
  int depth = 0;
  puzzleNodeQueue.push(puzzleNode);
  while (puzzleNodeQueue.size())
  {
    PUZZLE_NODE currentPuzzleNode = puzzleNodeQueue.front();
    if (checkObject(currentPuzzleNode, objPuzzleNode))
    {
      for (int i = 0; i < currentPuzzleNode.precedeActionList.size(); i++)
      {
        outputAction(currentPuzzleNode.precedeActionList[i], i + 1);
      }
      cout << "找到正确结果:" << endl;
      for (int i = 0; i < 3; i++)
      {
        cout << " " << currentPuzzleNode.puzzle[i * 3 + 0].puzzleId << "  " << currentPuzzleNode.puzzle[i * 3 + 1].puzzleId << "  " << currentPuzzleNode.puzzle[i * 3 + 2].puzzleId << endl;
      }
      cout << endl;

      result[0] = 1;
      result[1] = currentPuzzleNode.precedeActionList.size();
      return result;
    }
    else
    {
      visited[visitedNum(currentPuzzleNode)] = 1;
      if (currentPuzzleNode.nextActionList.size() == 0)
      {
        currentPuzzleNode = updatePuzzleNodeActionList(currentPuzzleNode);
      }
      puzzleNodeQueue.pop();
      for (int i = 0; i < currentPuzzleNode.nextActionList.size(); i++)
      {
        PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(currentPuzzleNode.nextActionList[i], currentPuzzleNode);
        if (!currentPuzzleNode.precedeActionList.empty())
        {
          for (int actionIndex = 0; actionIndex < currentPuzzleNode.precedeActionList.size(); actionIndex++)
          {
            nextPuzzleNode.precedeActionList.push_back(currentPuzzleNode.precedeActionList[actionIndex]);
          }
        }
        nextPuzzleNode.precedeActionList.push_back(currentPuzzleNode.nextActionList[i]);
        if (visited[visitedNum(nextPuzzleNode)] == 1)
        {
          continue;
        }
        nextPuzzleNode.depth = currentPuzzleNode.depth + 1;
        puzzleNodeQueue.push(nextPuzzleNode);
      }
    }
  }
  return result;
}

// 深度优先搜索
int *depthFirstSearch(PUZZLE_NODE initialNode, PUZZLE_NODE objPuzzleNode)
{

  // result[0] 1:correct;0:wrong
  // result[1] 步数 steps
  int result[2] = {0, 0};

  cout << "初始节点状态：" << endl;
  for (int i = 0; i < 3; i++)
  {
    cout << " " << initialNode.puzzle[i * 3 + 0].puzzleId << "  " << initialNode.puzzle[i * 3 + 1].puzzleId << "  " << initialNode.puzzle[i * 3 + 2].puzzleId << endl;
  }
  cout << endl;
  PUZZLE_NODE puzzleNode = initialNode;
  stack<PUZZLE_NODE> puzzleNodeStack;
  puzzleNode.depth = 0;
  int depth = 0;
  puzzleNodeStack.push(puzzleNode);
  while (puzzleNodeStack.size())
  {
    PUZZLE_NODE currentPuzzleNode = puzzleNodeStack.top();
    if (checkObject(currentPuzzleNode, objPuzzleNode))
    {

      for (int i = 0; i < currentPuzzleNode.precedeActionList.size(); i++)
      {
        outputAction(currentPuzzleNode.precedeActionList[i], i + 1);
      }
      cout << "找到正确结果:" << endl;
      for (int i = 0; i < 3; i++)
      {
        cout << " " << currentPuzzleNode.puzzle[i * 3 + 0].puzzleId << "  " << currentPuzzleNode.puzzle[i * 3 + 1].puzzleId << "  " << currentPuzzleNode.puzzle[i * 3 + 2].puzzleId << endl;
      }
      cout << endl;

      result[0] = 1;
      result[1] = currentPuzzleNode.nextActionList.size();
      return result;
    }
    else
    {
      visited[visitedNum(currentPuzzleNode)] = 1;
      if (currentPuzzleNode.nextActionList.size() == 0)
      {
        currentPuzzleNode = updatePuzzleNodeActionList(currentPuzzleNode);
      }
      puzzleNodeStack.pop();
      for (int i = 0; i < currentPuzzleNode.nextActionList.size(); i++)
      {
        PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(currentPuzzleNode.nextActionList[i], currentPuzzleNode);
        if (!currentPuzzleNode.precedeActionList.empty())
        {
          for (int actionIndex = 0; actionIndex < currentPuzzleNode.precedeActionList.size(); actionIndex++)
          {
            nextPuzzleNode.precedeActionList.push_back(currentPuzzleNode.precedeActionList[actionIndex]);
          }
        }
        nextPuzzleNode.precedeActionList.push_back(currentPuzzleNode.nextActionList[i]);
        if (visited[visitedNum(nextPuzzleNode)] == 1 || currentPuzzleNode.depth == 20)
        {
          continue;
        }
        nextPuzzleNode.depth = currentPuzzleNode.depth + 1;
        puzzleNodeStack.push(nextPuzzleNode);
      }
    }
  }
  return result;
}

// 启发式搜索1
int *heuristicSearchInformedByIncorrectNum(PUZZLE_NODE initialNode, PUZZLE_NODE objPuzzleNode)
{
  // result[0] 1:correct;0:wrong
  // result[1] 步数 steps
  int result[2] = {0, 0};
  cout << "初始节点状态：" << endl;
  for (int i = 0; i < 3; i++)
  {
    cout << " " << initialNode.puzzle[i * 3 + 0].puzzleId << "  " << initialNode.puzzle[i * 3 + 1].puzzleId << "  " << initialNode.puzzle[i * 3 + 2].puzzleId << endl;
  }
  cout << endl;
  PUZZLE_NODE puzzleNode = initialNode;
  puzzleNode.depth = 0;
  // 判断节点状态是否等于目标状态
  while (true)
  {
    if (checkObject(puzzleNode, objPuzzleNode))
    {
      for (int i = 0; i < puzzleNode.precedeActionList.size(); i++)
      {
        outputAction(puzzleNode.precedeActionList[i], i + 1);
      }
      cout << "找到正确结果:" << endl;
      for (int i = 0; i < 3; i++)
      {
        cout << " " << puzzleNode.puzzle[i * 3 + 0].puzzleId << "  " << puzzleNode.puzzle[i * 3 + 1].puzzleId << "  " << puzzleNode.puzzle[i * 3 + 2].puzzleId << endl;
      }
      cout << endl;

      result[0] = 1;
      result[1] = puzzleNode.depth;
      return result;
    }
    else
    {
      // 更新当前节点状态的动作列表
      puzzleNode = updatePuzzleNodeActionList(puzzleNode);
      vector<int> movement = InformedByIncorrectNum(puzzleNode, objPuzzleNode, visited);
      PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(movement, puzzleNode);
      visited[visitedNum(nextPuzzleNode)] = 1;
      nextPuzzleNode.depth = puzzleNode.depth + 1;
      if (!puzzleNode.precedeActionList.empty())
      {
        for (int actionIndex = 0; actionIndex < puzzleNode.precedeActionList.size(); actionIndex++)
        {
          nextPuzzleNode.precedeActionList.push_back(puzzleNode.precedeActionList[actionIndex]);
        }
      }
      nextPuzzleNode.precedeActionList.push_back(movement);
      puzzleNode = nextPuzzleNode;
    }
  }
  for (int i = 0; i < puzzleNode.precedeActionList.size(); i++)
  {
    outputAction(puzzleNode.precedeActionList[i], i + 1);
  }
  return result;
}

// 启发式搜素2
int *heuristicSearchInformedByManhattonDis(PUZZLE_NODE initialNode, PUZZLE_NODE objPuzzleNode)
{
  // result[0] 1:correct;0:wrong
  // result[1] 步数 steps
  int result[2] = {0, 0};
  cout << "初始节点状态：" << endl;
  for (int i = 0; i < 3; i++)
  {
    cout << " " << initialNode.puzzle[i * 3 + 0].puzzleId << "  " << initialNode.puzzle[i * 3 + 1].puzzleId << "  " << initialNode.puzzle[i * 3 + 2].puzzleId << endl;
  }
  cout << endl;
  PUZZLE_NODE puzzleNode = initialNode;
  puzzleNode.depth = 0;
  // 判断节点状态是否等于目标状态
  while (true)
  {
    if (checkObject(puzzleNode, objPuzzleNode))
    {
      for (int i = 0; i < puzzleNode.precedeActionList.size(); i++)
      {
        outputAction(puzzleNode.precedeActionList[i], i + 1);
      }
      cout << "找到正确结果:" << endl;
      for (int i = 0; i < 3; i++)
      {
        cout << " " << puzzleNode.puzzle[i * 3 + 0].puzzleId << "  " << puzzleNode.puzzle[i * 3 + 1].puzzleId << "  " << puzzleNode.puzzle[i * 3 + 2].puzzleId << endl;
      }
      cout << endl;

      result[0] = 1;
      result[1] = puzzleNode.depth;
      return result;
    }
    else
    {
      // 更新当前节点状态的动作列表
      puzzleNode = updatePuzzleNodeActionList(puzzleNode);
      vector<int> movement = InformedByManhattonDis(puzzleNode, objPuzzleNode, visited);
      PUZZLE_NODE nextPuzzleNode = moveToPuzzleNode(movement, puzzleNode);
      visited[visitedNum(nextPuzzleNode)] = 1;
      nextPuzzleNode.depth = puzzleNode.depth + 1;
      if (!puzzleNode.precedeActionList.empty())
      {
        for (int actionIndex = 0; actionIndex < puzzleNode.precedeActionList.size(); actionIndex++)
        {
          nextPuzzleNode.precedeActionList.push_back(puzzleNode.precedeActionList[actionIndex]);
        }
      }
      nextPuzzleNode.precedeActionList.push_back(movement);
      puzzleNode = nextPuzzleNode;
    }
  }
  for (int i = 0; i < puzzleNode.precedeActionList.size(); i++)
  {
    outputAction(puzzleNode.precedeActionList[i], i + 1);
  }
  return result;
}

int main()
{
    srand((unsigned)time(0));  
  PUZZLE_NODE objPuzzleNode;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      objPuzzleNode.puzzle[i * 3 + j].puzzleId = i * 3 + j;
      objPuzzleNode.puzzle[i * 3 + j].xPosition = i;
      objPuzzleNode.puzzle[i * 3 + j].yPosition = j;
    }
  }
  objPuzzleNode = updatePuzzleNodeActionList(objPuzzleNode);

  int setup = 0;
  while (setup != -1)
  {

    visited.clear();

    cout << "请输入调试设置(-1:退出;1:广度优先搜索;2:深度有限搜索;3:启发式搜索1;4:启发式搜索2):" << endl;
    cin >> setup;
    int backwardSteps;
    cout << "请输入大于等于5小于等于20的回退步数" << endl;
    cin >> backwardSteps;
    while (backwardSteps < 5 || backwardSteps > 20)
    {
      cout << "输入错误,请输入大于等于5小于等于20的回退步数" << endl;
      cin >> backwardSteps;
    }

    PUZZLE_NODE initialNode = initialPuzzleNode(backwardSteps);

    int *result;
    if (setup == 1)
    {
      result = binaryFirstSearch(initialNode, objPuzzleNode);
    }
    else if (setup == 2)
    {
      result = depthFirstSearch(initialNode, objPuzzleNode);
    }
    else if (setup == 3)
    {
      result = heuristicSearchInformedByIncorrectNum(initialNode, objPuzzleNode);
    }
    else if (setup == 4)
    {
      result = heuristicSearchInformedByManhattonDis(initialNode, objPuzzleNode);
    }
    else
    {
      cout << "输入设置有误，请重新运行" << endl;
      return 0;
    }

    if (result[0] == 1)
    {
      cout << "结果为correct" << endl;
    }
    else
    {
      cout << "结果为wrong" << endl;
    }
  }
  return 0;
}