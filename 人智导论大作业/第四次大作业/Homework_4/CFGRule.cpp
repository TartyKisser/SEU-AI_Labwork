#include "CFGRule.hpp"

CFGRule::CFGRule(string lhs, vector<string> rhs, double prob)
{
    this->lhs = lhs;
    this->rhs = rhs;
    this->prob = prob;
}
