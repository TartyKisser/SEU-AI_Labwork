#include "LexRule.hpp"

LexRule::LexRule(string pos, string lexicon, double prob)
{
    this->pos = pos;
    this->lexicon = lexicon;
    this->prob = prob;
}
