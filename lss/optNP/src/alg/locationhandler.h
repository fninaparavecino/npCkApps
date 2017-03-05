/*
 * Northeastern University
 * Computer Architecture Research Group
 * NUCAR
 */

#ifndef LOCHANDLER_H
#define LOCHANDLER_H

#include <stdio.h>
#include <stdlib.h>
#include <climits>

#include "../config.h"

using namespace std;

unsigned int imageLocation (unsigned int x,
		unsigned int y,
		unsigned int gridXSize);

#endif
